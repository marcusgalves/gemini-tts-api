# main.py

import base64 # Adicionado
import io
import mimetypes # Embora não usado diretamente no novo endpoint, mantido por consistência
import struct # Para a função convert_to_wav

from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator # field_validator é para Pydantic V2+

# Importações da biblioteca Gemini
from google import genai
from google.genai import types

# --- Funções auxiliares migradas do app.py ---
def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Analisa bits por amostra e taxa de um string de tipo MIME de áudio.

    Assume que bits por amostra é codificado como "L16" e taxa como "rate=xxxxx".

    Args:
        mime_type: O string do tipo MIME de áudio (ex: "audio/L16;rate=24000").

    Returns:
        Um dicionário com chaves "bits_per_sample" e "rate". Os valores serão
        inteiros se encontrados, caso contrário, padrões são usados.
    """
    bits_per_sample = 16  # Padrão
    rate = 24000          # Padrão

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass  # Mantém o valor padrão se a extração falhar
        # Modificado para pegar o Lxx da parte principal do mime_type também
        elif param.lower().startswith("audio/l"): 
            type_part_potential = param.split('/', 1)[-1] # Ex: L16 ou L16;codec=pcm
            # Pega somente a parte antes de um possível ';'
            actual_type_part = type_part_potential.split(';',1)[0] # Ex: L16
            if actual_type_part.lower().startswith("l"):
                try:
                    bits_str = actual_type_part[1:]
                    bits_per_sample = int(bits_str)
                except (ValueError, IndexError):
                    pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Gera um cabeçalho de arquivo WAV para os dados de áudio e parâmetros fornecidos.

    Args:
        audio_data: Os dados de áudio brutos como um objeto bytes.
        mime_type: Tipo MIME dos dados de áudio brutos (ex: "audio/L16;rate=24000").

    Returns:
        Um objeto bytes representando o arquivo WAV completo (cabeçalho + dados).
    """
    parameters = parse_audio_mime_type(mime_type)
    # Garante que temos valores padrão se não forem encontrados no parse
    bits_per_sample = parameters.get("bits_per_sample")
    if bits_per_sample is None: bits_per_sample = 16 

    sample_rate = parameters.get("rate")
    if sample_rate is None: sample_rate = 24000
    
    num_channels = 1  # Áudio mono
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # Subchunk1Size (16 para PCM)
        1,  # AudioFormat (1 para PCM)
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + audio_data

# --- Lógica de Geração de Áudio com Gemini ---
def generate_audio_from_gemini(
    api_key: str, text: str, voice: str, temperature: float, model_name: str
) -> bytes | None:
    """
    Gera áudio usando a API Gemini e retorna os bytes do arquivo WAV.
    """
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Falha ao inicializar o cliente Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"Falha na configuração do cliente Gemini: {e}")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )

    raw_audio_chunks = []
    source_mime_type = None

    try:
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or not chunk.candidates[0].content
                or not chunk.candidates[0].content.parts
            ):
                continue
            
            part = chunk.candidates[0].content.parts[0]
            if part.inline_data and part.inline_data.data:
                raw_audio_chunks.append(part.inline_data.data)
                if source_mime_type is None: # Captura o mime_type do primeiro chunk com dados
                    source_mime_type = part.inline_data.mime_type
    except Exception as e:
        print(f"Erro durante a chamada à API Gemini: {e}")
        raise HTTPException(status_code=502, detail=f"Erro na comunicação com a API Gemini: {e}")

    if not raw_audio_chunks: # Não há chunks de áudio, source_mime_type pode ser None
        print("Nenhum chunk de áudio recebido do Gemini.")
        return None
    
    # Se source_mime_type não foi definido, mas temos chunks, precisamos de um fallback ou erro.
    # Para este exemplo, vamos assumir um mime_type padrão se não for pego, embora seja melhor garantir que ele seja definido.
    if source_mime_type is None and raw_audio_chunks:
        print("AVISO: source_mime_type não foi detectado, usando padrão para conversão.")
        source_mime_type = "audio/L16;rate=24000" # Ou levante um erro

    concatenated_raw_audio = b"".join(raw_audio_chunks)
    
    if source_mime_type and source_mime_type.startswith("audio/wav"):
        return concatenated_raw_audio
    elif source_mime_type: 
        print(f"Convertendo de {source_mime_type} para WAV.")
        return convert_to_wav(concatenated_raw_audio, source_mime_type)
    else: 
        print("Mime type da fonte desconhecido, não foi possível converter para WAV.")
        return None


# --- Configuração da API FastAPI ---
app = FastAPI(
    title="API de Geração de Áudio com Gemini",
    description="Esta API permite gerar áudio a partir de texto usando o modelo Gemini TTS e converter áudio Base64 para WAV.",
    version="1.1.0" # Versão incrementada
)

VOICES = [
    {"name": "Zephyr", "style": "Bright"}, {"name": "Puck", "style": "Upbeat"},
    {"name": "Charon", "style": "Informative"}, {"name": "Kore", "style": "Firm"},
    {"name": "Fenrir", "style": "Excitable"}, {"name": "Leda", "style": "Youthful"},
    {"name": "Orus", "style": "Firm"}, {"name": "Aoede", "style": "Breezy"},
    {"name": "Callirrhoe", "style": "Easy-going"}, {"name": "Autonoe", "style": "Bright"},
    {"name": "Enceladus", "style": "Breathy"}, {"name": "Iapetus", "style": "Clear"},
    {"name": "Umbriel", "style": "Easy-going"}, {"name": "Algieba", "style": "Smooth"},
    {"name": "Despina", "style": "Smooth"}, {"name": "Erinome", "style": "Clear"},
    {"name": "Algenib", "style": "Gravelly"}, {"name": "Rasalgethi", "style": "Informative"},
    {"name": "Laomedeia", "style": "Upbeat"}, {"name": "Achernar", "style": "Soft"},
    {"name": "Alnilam", "style": "Firm"}, {"name": "Schedar", "style": "Even"},
    {"name": "Gacrux", "style": "Mature"}, {"name": "Pulcherrima", "style": "Forward"},
    {"name": "Achird", "style": "Friendly"}, {"name": "Zubenelgenubi", "style": "Casual"},
    {"name": "Vindemiatrix", "style": "Gentle"}, {"name": "Sadachbia", "style": "Lively"},
    {"name": "Sadaltager", "style": "Knowledgeable"}, {"name": "Sulafat", "style": "Warm"},
]

@app.get("/voices", tags=["Voices"])
async def get_voices_endpoint():
    """Retorna a lista de vozes disponíveis."""
    return VOICES

# Pydantic Model para o corpo da requisição (sem a api_key)
class GenerateBodyParams(BaseModel):
    text: str = Body(..., description="O texto a ser convertido em áudio.", example="Olá! Esse aqui é um teste de geração de voz em português brasileiro!")
    voice: str = Body(..., description="O nome da voz a ser usada.", example="Zephyr")
    temperature: float = Body(1.0, description="A temperatura para a geração (controla a aleatoriedade).", ge=0.0, le=2.0)
    model: str = Body("gemini-2.5-flash-preview-tts", description="O modelo Gemini TTS a ser usado.")

    @field_validator("voice")
    def validate_voice(cls, v_value):
        allowed_voices = [vox["name"] for vox in VOICES]
        if v_value not in allowed_voices:
            raise ValueError(f"A voz deve ser uma das seguintes: {allowed_voices}")
        return v_value

# Pydantic Model para o novo endpoint /convert
class ConvertAudioBody(BaseModel):
    base64_audio: str = Body(..., description="String Base64 do áudio a ser convertido.")
    mime_type: str = Body(..., description="MIME type original do áudio (ex: 'audio/L16;rate=24000').", example="audio/L16;rate=24000")


# Endpoint modificado para aceitar API key como Query Parameter
@app.post("/generate-audio", tags=["Audio Generation"])
async def generate_audio_endpoint(
    body_params: GenerateBodyParams, 
    api_key_from_query: str = Query(..., alias="key", description="Sua chave da API Gemini passada como query parameter 'key'.") 
):
    """
    Gera áudio a partir do texto fornecido usando a voz e configurações especificadas.
    A credencial da API Gemini deve ser fornecida como um query parameter chamado 'key'.
    """
    audio_bytes: bytes | None

    try:
        audio_bytes = generate_audio_from_gemini(
            api_key=api_key_from_query,
            text=body_params.text,
            voice=body_params.voice,
            temperature=body_params.temperature,
            model_name=body_params.model,
        )
    except HTTPException:
        raise 
    except Exception as e:
        print(f"Erro inesperado ao gerar áudio: {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar a solicitação: {e}")

    if not audio_bytes:
        raise HTTPException(
            status_code=500, detail="Nenhum dado de áudio foi retornado pelo serviço Gemini ou a conversão falhou."
        )

    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

# NOVO ENDPOINT /convert
@app.post("/convert", tags=["Audio Conversion"])
async def convert_audio_endpoint(payload: ConvertAudioBody):
    """
    Recebe áudio em Base64 e seu MIME type original, e o converte para formato WAV.
    """
    try:
        raw_audio_data = base64.b64decode(payload.base64_audio)
    except base64.binascii.Error as e: # Erro específico para base64 inválido
        raise HTTPException(status_code=400, detail=f"Formato Base64 inválido: {e}")
    except Exception as e: # Outros erros de decodificação
        raise HTTPException(status_code=400, detail=f"Erro ao decodificar Base64: {e}")

    if not raw_audio_data:
        raise HTTPException(status_code=400, detail="Dados de áudio em Base64 resultaram em bytes vazios.")

    try:
        wav_audio_bytes = convert_to_wav(raw_audio_data, payload.mime_type)
    except Exception as e:
        print(f"Erro durante a conversão para WAV: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao converter áudio para WAV: {e}")

    if not wav_audio_bytes: # Checagem extra, embora convert_to_wav deva sempre retornar bytes ou falhar
        raise HTTPException(
            status_code=500, detail="Conversão para WAV não retornou dados."
        )

    return StreamingResponse(io.BytesIO(wav_audio_bytes), media_type="audio/wav")


# Para rodar localmente com Uvicorn:
# uvicorn main:app --reload
#
# Exemplo de requisição para /convert:
# POST http://127.0.0.1:8000/convert
# Corpo (Body) da requisição (JSON):
# {
#   "base64_audio": "SUQzBAAAAAAA...", // Sua string base64 aqui (este é um exemplo de MP3, use o seu L16)
#   "mime_type": "audio/L16;rate=24000"
# }
