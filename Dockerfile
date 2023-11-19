# FROM ubuntu:23.10
FROM python:3.9.17
COPY ./api /
COPY models/svc.pkl /
WORKDIR /
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENV NAME World
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
RUN pytest -v


