import base64
import io
import mimetypes
import struct
import os
import httpx

from fastapi import FastAPI, HTTPException, Body, Query, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

# Importações da biblioteca Gemini
from google import genai
from google.genai import types

# --- Funções auxiliares (sem alterações) ---
def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Analisa bits por amostra e taxa de um string de tipo MIME de áudio."""
    bits_per_sample = 16
    rate = 24000

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.lower().startswith("audio/l"):
            type_part_potential = param.split('/', 1)[-1]
            actual_type_part = type_part_potential.split(';', 1)[0]
            if actual_type_part.lower().startswith("l"):
                try:
                    bits_str = actual_type_part[1:]
                    bits_per_sample = int(bits_str)
                except (ValueError, IndexError):
                    pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Gera um cabeçalho de arquivo WAV para os dados de áudio e parâmetros fornecidos."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters.get("bits_per_sample", 16)
    sample_rate = parameters.get("rate", 24000)

    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ",
        16, 1, num_channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b"data", data_size,
    )
    return header + audio_data


# --- Lógica de Geração de Áudio com Gemini (Corrigida) ---
def generate_audio_from_gemini(
    api_key: str, text: str, voice: str, language_code: str | None, temperature: float, model_name: str, proxy_url: str | None = None
) -> bytes | None:
    """
    Gera áudio usando a API Gemini. Utiliza um proxy e language_code se fornecidos.
    """
    original_https_proxy = os.environ.get('HTTPS_PROXY')
    original_http_proxy = os.environ.get('HTTP_PROXY')

    if proxy_url:
        os.environ['HTTPS_PROXY'] = proxy_url
        os.environ['HTTP_PROXY'] = proxy_url

    try:
        try:
            client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"Falha ao inicializar o cliente Gemini: {e}")
            raise HTTPException(status_code=500, detail=f"Falha na configuração do cliente Gemini: {e}")

        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=text)]),
        ]

        # **CORREÇÃO 1: Usar dicionários para safety_settings (formato mais compatível)**
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]

        # **CORREÇÃO 2: Constrói o SpeechConfig dinamicamente**
        speech_config_args = {
            "voice_config": types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        }
        if language_code:
            speech_config_args["language_code"] = language_code
        
        speech_config = types.SpeechConfig(**speech_config_args)

        # **CORREÇÃO 3: Usar string para response_modalities**
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            response_modalities=["AUDIO"],
            speech_config=speech_config,
            safety_settings=safety_settings
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
                    if source_mime_type is None:
                        source_mime_type = part.inline_data.mime_type
        except Exception as api_error:
            # **CORREÇÃO 4: Capturar e retornar erros específicos da API Gemini**
            print(f"Erro específico da API Gemini: {api_error}")
            error_msg = str(api_error)
            
            # Verificar se é um erro de autenticação
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise HTTPException(status_code=401, detail=f"Erro de autenticação na API Gemini: {error_msg}")
            # Verificar se é um erro de quota/limite
            elif "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise HTTPException(status_code=429, detail=f"Limite de quota/rate limit da API Gemini: {error_msg}")
            # Verificar se é um erro de modelo não encontrado
            elif "404" in error_msg or "not found" in error_msg.lower():
                raise HTTPException(status_code=404, detail=f"Modelo não encontrado na API Gemini: {error_msg}")
            # Outros erros da API
            else:
                raise HTTPException(status_code=502, detail=f"Erro da API Gemini: {error_msg}")

    except httpx.ProxyError as e:
        print(f"Falha na conexão com o proxy: {e}")
        raise HTTPException(status_code=400, detail=f"Falha ao conectar ao proxy '{proxy_url}'. Verifique o URL e a disponibilidade do proxy. Erro: {e}")
    except HTTPException:
        # Re-raise HTTPExceptions para não mascarar os erros específicos
        raise
    except Exception as e:
        print(f"Erro durante a chamada à API Gemini: {e}")
        raise HTTPException(status_code=502, detail=f"Erro na comunicação com a API Gemini: {e}")
    finally:
        if original_https_proxy:
            os.environ['HTTPS_PROXY'] = original_https_proxy
        elif 'HTTPS_PROXY' in os.environ:
            del os.environ['HTTPS_PROXY']

        if original_http_proxy:
            os.environ['HTTP_PROXY'] = original_http_proxy
        elif 'HTTP_PROXY' in os.environ:
            del os.environ['HTTP_PROXY']

    if not raw_audio_chunks:
        print("Nenhum chunk de áudio recebido do Gemini.")
        return None

    if source_mime_type is None and raw_audio_chunks:
        print("AVISO: source_mime_type não foi detectado, usando padrão para conversão.")
        source_mime_type = "audio/L16;rate=24000"

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
    version="1.5.1"
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

class GenerateBodyParams(BaseModel):
    text: str = Body(..., description="O texto a ser convertido em áudio.", example="Olá! Esse aqui é um teste de geração de voz em português brasileiro!")
    voice: str = Body(..., description="O nome da voz a ser usada.", example="Zephyr")
    language_code: str | None = Body(None, description="Código de idioma opcional para a geração (ex: 'pt-BR', 'en-US').", example="pt-BR")
    temperature: float = Body(1.0, description="A temperatura para a geração (controla a aleatoriedade).", ge=0.0, le=2.0)
    model: str = Body("gemini-2.5-flash-preview-tts", description="O modelo Gemini TTS a ser usado.")

    @field_validator("voice")
    def validate_voice(cls, v_value):
        allowed_voices = [vox["name"] for vox in VOICES]
        if v_value not in allowed_voices:
            raise ValueError(f"A voz deve ser uma das seguintes: {allowed_voices}")
        return v_value

class ConvertAudioBody(BaseModel):
    base64_audio: str = Body(..., description="String Base64 do áudio a ser convertido.")
    mime_type: str = Body(..., description="MIME type original do áudio (ex: 'audio/L16;rate=24000').", example="audio/L16;rate=24000")


@app.post("/generate-audio", tags=["Audio Generation"])
async def generate_audio_endpoint(
    body_params: GenerateBodyParams,
    api_key_from_query: str = Query(..., alias="key", description="Sua chave da API Gemini passada como query parameter 'key'."),
    proxy_url: str | None = Header(
        None,
        description="URL do proxy opcional no formato 'PROTOCOL://USERNAME:PASSWORD@IP:PORT'. Suporta http, https e socks5.",
        alias="proxy_url"
    )
):
    """
    Gera áudio a partir do texto fornecido.
    A credencial da API Gemini deve ser fornecida via query param 'key'.
    Pode-se utilizar um proxy para a requisição enviando o header 'proxy_url'.
    """
    audio_bytes: bytes | None

    try:
        audio_bytes = generate_audio_from_gemini(
            api_key=api_key_from_query,
            text=body_params.text,
            voice=body_params.voice,
            language_code=body_params.language_code,
            temperature=body_params.temperature,
            model_name=body_params.model,
            proxy_url=proxy_url
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


@app.post("/convert", tags=["Audio Conversion"])
async def convert_audio_endpoint(payload: ConvertAudioBody):
    """
    Recebe áudio em Base64 e seu MIME type original, e o converte para formato WAV.
    """
    try:
        raw_audio_data = base64.b64decode(payload.base64_audio)
    except base64.binascii.Error as e:
        raise HTTPException(status_code=400, detail=f"Formato Base64 inválido: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao decodificar Base64: {e}")

    if not raw_audio_data:
        raise HTTPException(status_code=400, detail="Dados de áudio em Base64 resultaram em bytes vazios.")

    try:
        wav_audio_bytes = convert_to_wav(raw_audio_data, payload.mime_type)
    except Exception as e:
        print(f"Erro durante a conversão para WAV: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao converter áudio para WAV: {e}")

    if not wav_audio_bytes:
        raise HTTPException(
            status_code=500, detail="Conversão para WAV não retornou dados."
        )

    return StreamingResponse(io.BytesIO(wav_audio_bytes), media_type="audio/wav")
