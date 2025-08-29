from flask import Flask, render_template, request, jsonify, g
import requests
import json
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)

#function to get db
DATABASE = 'ThotRelay.db'

# Ollama API configuration
OLLAMA_HOST = "http://localhost:11434"
AVAILABLE_MODELS = ["gpt-oss:20b", "deepseek-r1:8b", "llama3:8b"]  # Updated model list

#define function
def get_db():
    db = getattr(g, '_database',None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE);
    return db;

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database',None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        model = data.get('model', 'gpt-oss:20b')  # Changed default model
        prompt = data.get('prompt', '')
        
        # Prepare the request to Ollama
        ollama_request = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Send request to Ollama
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=ollama_request,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "success": True,
                "response": result.get('response', 'No response generated')
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Ollama API error: {response.status_code}"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)