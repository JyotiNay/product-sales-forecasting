web: bash -c "python deployment/train_server.py && gunicorn deployment.app:app --preload --bind 0.0.0.0:$PORT"

