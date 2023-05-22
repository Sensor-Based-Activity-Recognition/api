docker build . -t sensor-based-activity-recognition-api
docker run -p 8050:8050 --name sensor-based-activity-recognition-api --network nginxproxymanager_default sensor-based-activity-recognition-api 