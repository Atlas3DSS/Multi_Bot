
from flask import Flask
import os
from waitress import serve
import logging

def create_app(character_name):
    app = Flask(character_name)
    
    @app.route('/')
    def home():
        return f"Hello. I am {character_name} and I am alive!"
    
    return app

def run_server(character_name):
    app = create_app(character_name)
    primary_port = int(os.environ.get('PORT', 8080))
    
    print(f"Starting {character_name} server on port {primary_port}")
    try:
        serve(app, host='0.0.0.0', port=primary_port)
    except Exception as e:
        print(f"Error starting {character_name} server on primary port: {e}")
        try:
            # Try the external port 80 as specified in .replit
            print("Attempting to start on port 80")
            serve(app, host='0.0.0.0', port=80)
        except Exception as e:
            print(f"Error starting on port 80: {e}")
            # Final fallback
            alt_port = primary_port + 1000
            print(f"Final attempt on port {alt_port}")
            serve(app, host='0.0.0.0', port=alt_port)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_server(sys.argv[1])
    else:
        print("Please provide a character name as an argument.")
