# gunicorn_config/gunicorn_config_pdf_query.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../piazza_query_api')))

bind = '0.0.0.0:7001'
workers = 1  # Number of worker processes
threads = 1  # Number of threads per worker
timeout = 120  # Timeout in seconds
loglevel = 'info'  # Logging level
accesslog = '-'  # Log to stdout (useful for PM2)
errorlog = '-'  # Log to stdout (useful for PM2)