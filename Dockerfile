# Usa Python 3.11 slim (leve e estável)
FROM python:3.11-slim

# Define diretório de trabalho
WORKDIR /app

# Copia requirements primeiro (para cache)
COPY requirements.txt .

# Instala dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código
COPY . .

# Expõe porta (padrão do Flask)
EXPOSE 8080

# Comando para rodar o app
CMD ["python", "app.py"]
