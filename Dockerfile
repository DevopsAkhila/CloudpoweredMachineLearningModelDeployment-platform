FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mlserve ./mlserve
COPY README.md .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "-m", "mlserve.app"]
