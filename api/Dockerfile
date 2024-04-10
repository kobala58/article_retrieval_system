FROM python:3.11
LABEL authors="kobala"

WORKDIR /code
COPY requirements.txt requirements.txt
RUN apt-get update
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python3", "-m" ,"uvicorn", "retrieval:app", "--host", "0.0.0.0", "--port", "8000"]
