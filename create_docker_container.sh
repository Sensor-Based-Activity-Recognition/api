docker build . -t sensor-based-activity-recognition-api
docker run -p 8098:8098 --name sensor-based-activity-recognition-api sensor-based-activity-recognition-api