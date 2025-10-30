import os
bind = f":{os.getenv('PORT', '5001')}"
workers = 1  # TensorFlow + Windows/Containers: stick to one worker
threads = 2
timeout = 300
graceful_timeout = 60
preload_app = False
worker_class = "gthread"
loglevel = "info"
