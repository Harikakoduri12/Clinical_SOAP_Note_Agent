# Clinical SOAP Note Agent
# Build:  docker build -t soap-agent .
# Run:    docker run -p 8000:8000 -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY soap-agent

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "soap_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
