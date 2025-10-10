from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Ollama API config
OLLAMA_HOST = "http://localhost:11434"
AVAILABLE_MODELS = ["phi3:mini", "llama3:8b-q4", "gemma:2b", "deepseek-r1"]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/about')
def about():
    slides = [
        'slide1.jpg',
        'slide2.jpg',
        'slide3.jpg'
    ]
    return render_template('about.html', slides=slides)


@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/payment')
def payment():
    return render_template('payment.html')

@app.route('/nmap')
def nmap():
    return render_template('nmap.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        model = data.get('model', 'phi3:mini')
        prompt = data.get('prompt', '')

        if model == 'deepseek-r1':
            ai_response = f"[DeepSeek simulated response to: {prompt}]"
        else:
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
