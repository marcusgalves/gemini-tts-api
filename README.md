# Gemini TTS API

## Overview

This project is a FastAPI application that provides Text-to-Speech (TTS) functionality using Google's Gemini API. It allows users to convert text into audio by selecting from a list of available voices and leveraging the powerful multimodal capabilities of Gemini.

The API is designed to be containerized using Docker for easy deployment and scaling, particularly with Docker Swarm.

## API Endpoints

### 1. Get Available Voices

* **Endpoint:** `GET /voices`
* **Description:** Retrieves a list of available prebuilt voices that can be used for TTS generation.
* **Method:** `GET`
* **Successful Response:** `200 OK`
    * **Content-Type:** `application/json`
    * **Body Example:**
        ```json
        [
          {"name": "Zephyr", "style": "Bright"},
          {"name": "Puck", "style": "Upbeat"},
          // ... other voices listed in your main.py
        ]
        ```

### 2. Generate Audio

* **Endpoint:** `POST /generate-audio`
* **Description:** Generates audio from the provided text using the specified voice, model, temperature, and your Gemini API key.
* **Method:** `POST`
* **Query Parameters:**
    * `key` (string, **required**): Your Google Gemini API key.
* **Request Body:**
    * **Content-Type:** `application/json`
    * **Body Structure:**
        ```json
        {
          "text": "Hello, this is a test message to be converted to speech.",
          "voice": "Zephyr",
          "temperature": 1.0,
          "model": "gemini-2.5-flash-preview-tts"
        }
        ```
        * `text` (string, **required**): The text content to be synthesized into speech.
        * `voice` (string, **required**): The name of the voice to use (e.g., "Zephyr"). Must be one of the voices returned by the `GET /voices` endpoint.
        * `temperature` (float, optional, default: `1.0`): Controls the randomness of the generation.
        * `model` (string, optional, default: `"gemini-2.5-flash-preview-tts"`): The specific Gemini TTS model to use.
* **Successful Response:**
    * Status: `200 OK`
    * Content-Type: `audio/wav`
    * Body: The binary audio data in WAV format.
* **Error Responses:**
    * `400 Bad Request`: Invalid input, such as missing required fields or an invalid voice name.
    * `404 Not Found`: If the endpoint path is incorrect.
    * `500 Internal Server Error`: If there's an issue on the server-side during audio generation or with the Gemini API client setup.
    * `502 Bad Gateway`: If the API encounters an error while communicating with the upstream Gemini API.

## Running the API

The API is intended to be run as a Docker container. Ensure your application uses port `8698`.

### Prerequisites

* Python 3.11+ (Python 3.13 was used during development)
* Docker Engine and Docker Compose (or Docker Desktop)

### Using Docker (Recommended)

#### 1. Building the Docker Image (if not pre-built)

If you need to build the image from the `Dockerfile`:
1.  Ensure `Dockerfile`, `main.py`, and `requirements.txt` are in your project root.
2.  Navigate to the project root in your terminal.
3.  Build the image:
    ```bash
    docker build -t your-gemini-tts-api-image:latest .
    ```
    (Replace `your-gemini-tts-api-image:latest` with your desired image name and tag).

#### 2. Running with Docker CLI (for local testing)

Once the image is built:
```bash
docker run -d -p 8698:8698 --name gemini_tts_api_instance your-gemini-tts-api-image:latest