FROM python:3.10-slim

WORKDIR /app

# copy project
COPY . /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "-m", "src.api_app"]
