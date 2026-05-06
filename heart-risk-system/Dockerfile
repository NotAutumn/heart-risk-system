FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python run_pipeline.py

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "--threads", "8", "-b", "0.0.0.0:5000", "app:app"]
