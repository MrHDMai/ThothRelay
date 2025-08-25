from flask import Flask
import redis

app = Flask(__name__)

@app.route('/')
# def hello():
#     try:
#         r = redis.Redis(host='localhost', port=6379)
#         return f"Redis says: {r.ping()}"
#     except redis.ConnectionError:
#         return "Cannot connect to Redis. Is it running?"



if __name__ == '__main__':
    app.run(debug=True)
