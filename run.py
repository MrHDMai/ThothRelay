from flask import Flask, render_template, request, jsonify, g
import requests
import json
import psycopg2
import os
from datetime import datetime
from psycopg2.extras import RealDictCursor

# Import the DeepSeek model
from app.model.deepseek_utils import deepseek_model

app = Flask(__name__)

# PostgreSQL configuration
DATABASE_CONFIG = {
    'dbname': 'ollama_chat',
    'user': 'ollama_user',
    'password': 'your_secure_password',
    'host': 'localhost',
    'port': '5432'
}

# Ollama API configuration
OLLAMA_HOST = "http://localhost:11434"
AVAILABLE_MODELS = ["phi3:mini", "llama3:8b-q4", "gemma:2b", "deepseek-r1"]

# Database functions
def get_db():
    if 'db' not in g:
        g.db = psycopg2.connect(**DATABASE_CONFIG)
    return g.db

@app.teardown_appcontext
def close_connection(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    # Connect to the default database to create our database if needed
    conn = psycopg2.connect(
        dbname='postgres',
        user=DATABASE_CONFIG['user'],
        password=DATABASE_CONFIG['password'],
        host=DATABASE_CONFIG['host'],
        port=DATABASE_CONFIG['port']
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Check if database exists, create if not
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'ollama_chat'")
    exists = cursor.fetchone()
    if not exists:
        cursor.execute('CREATE DATABASE ollama_chat')
    
    cursor.close()
    conn.close()
    
    # Now connect to our database and create tables
    conn = psycopg2.connect(**DATABASE_CONFIG)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            model_used TEXT,
            user_message TEXT,
            ai_response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        model = data.get('model', 'phi3:mini')
        prompt = data.get('prompt', '')
        
        # Check if we're using DeepSeek
        if model == 'deepseek-r1':
            # Use the local DeepSeek model
            ai_response = deepseek_model.generate_response(prompt)
        else:
            # Use Ollama for other models
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
                ai_response = result.get('response', 'No response generated')
            else:
                return jsonify({
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}"
                })
        
        # Save to database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (model_used, user_message, ai_response) VALUES (%s, %s, %s)",
            (model, prompt, ai_response)
        )
        conn.commit()
        
        return jsonify({
            "success": True,
            "response": ai_response
        })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        })

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        conn = get_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM conversations ORDER BY timestamp DESC LIMIT 50")
        conversations = cursor.fetchall()
        
        return jsonify({'success': True, 'history': conversations})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Initialize database on first run
    init_db()
    app.run(debug=True, port=5000)