FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV SOUNDDET_SERVICE_HOST=0.0.0.0
EXPOSE 8765
CMD ["python", "scripts/run_service.py"]
