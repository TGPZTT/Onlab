FROM python:3.11.6-bookworm

WORKDIR /app

COPY . /app

RUN pip install -r req.txt

CMD ["python", "hungarian_sampling.py"]
