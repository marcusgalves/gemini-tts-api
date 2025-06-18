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


# --- Lógica de Geração de Áudio com Gemini (Modificada para Safety Settings) ---
def generate_audio_from_gemini(
    api_key: str, text: str, voice: str, temperature: float, model_name: str, proxy_url: str | None = None
) -> bytes | None:
    """
    Gera áudio usando a API Gemini. Utiliza um proxy se a URL for fornecida via header.
    As configurações de segurança são desativadas por padrão.
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
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )

        # Desativa todas as configurações de segurança do Gemini
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
        ]

        raw_audio_chunks = []
        source_mime_type = None

        # ATUALIZADO: Adicionado o parâmetro safety_settings na chamada
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
            safety_settings=safety_settings  # Parâmetro adicionado aqui
        ):
            if (
                chunk.candidates is None
                or not chunk.candidates[0].content
                or not chunk.candidates[0].
