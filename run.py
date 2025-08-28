from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

# Ollama API configuration
OLLAMA_HOST = "http://localhost:11434"
AVAILABLE_MODELS = ["gpt-oss:20b", "deepseek-r1:8b", "llama3:8b"]  # Updated model list

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