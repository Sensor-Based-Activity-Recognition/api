
FROM python:3.10-bullseye

RUN mkdir -p /app
WORKDIR /app
COPY . /app

EXPOSE 8098
CMD sh -v -x run_server.sh