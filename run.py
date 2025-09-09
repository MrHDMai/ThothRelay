from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Ollama API configuration
OLLAMA_HOST = "http://localhost:11434"
AVAILABLE_MODELS = ["phi3:mini", "llama3:8b-q4", "gemma:2b", "deepseek-r1"]

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/about')
def about():
    # List of images for the slideshow
    slides = [
        'slide1.jpg',
        'slide2.jpg',
        'slide3.jpg'
    ]
    return render_template('about.html', slides=slides)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        model = data.get('model', 'phi3:mini')
        prompt = data.get('prompt', '')

        # Handle DeepSeek locally (placeholder)
        if model == 'deepseek-r1':
            ai_response = f"[DeepSeek simulated response to: {prompt}]"
        else:
            # Use Ollama for other models
            ollama_request = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }

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

        return jsonify({
            "success": True,
            "response": ai_response
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
