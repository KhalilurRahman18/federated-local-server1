from flask import Flask
app = Flask(__name__)

@app.get('/api')
def hello_world():
    return "Hello, World!"

