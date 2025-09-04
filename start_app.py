#!/usr/bin/env python3
"""
Start script for Thoth Relay application
"""
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the application
from run import app

if __name__ == '__main__':
    print("Starting Thoth Relay server...")
    print("Available models: phi3:mini, llama3:8b-q4, gemma:2b, deepseek-r1")
    app.run(debug=True, host='0.0.0.0', port=5000)