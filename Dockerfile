FROM python:3.10

WORKDIR /usr/wifi

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "wifi.py"]