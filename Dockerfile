# Imagem oficial do Python como imagem base
FROM python:3.13-slim

# Todos os comandos subsequentes serão executados a partir deste diretório
WORKDIR /app

# Se requirements.txt não mudar, o Docker reutilizará a camada da instalação das dependências.
COPY requirements.txt .

# Instalar pip mais recente e as dependências do projeto
# --no-cache-dir reduz o tamanho da imagem ao não armazenar o cache do pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


COPY main.py .

EXPOSE 8698

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8698"]
