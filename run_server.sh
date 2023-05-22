pip install -r requirements.txt
gunicorn api.wsgi -b :8050