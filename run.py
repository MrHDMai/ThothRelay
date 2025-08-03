# app.py
from flask import Flask
import redis
app = Flask(__name__)

@app.route('/')
def hello():
    r = redis.Redis(host='localhost', port=6379)
    print(r.ping())

if __name__ == '__main__':
    app.run(debug=True)
