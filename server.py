from flask import Flask, abort
import os
from waitress import serve
import logging
import threading
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate environment variables
required_env_vars = [
    'ANTHROPIC_API_KEY', 'CLAUDE_BOT_TOKEN', 'TOMMY_BOT_TOKEN', 
    'BOB_BOT_TOKEN', 'GEORGE_BOT_TOKEN', 'SYBIL_BOT_TOKEN'
]

for var in required_env_vars:
    if not os.environ.get(var):
        logger.warning(f"Missing environment variable: {var}")

VALID_BOTS = ['tommy', 'bob', 'claude', 'george', 'sybil']

def create_app():
    app = Flask("bot_server")

    @app.route('/<bot_name>')
    def bot_route(bot_name):
        if bot_name.lower() not in VALID_BOTS:
            abort(404)
        return f"Hello. I am {bot_name} and I am alive!"

    @app.route('/')
    def home():
        return "Bot server running! All systems operational.", 200

    @app.route('/health')
    def health_check():
        return "OK", 200

    return app

def run_server():
    """Run the Flask server with waitress"""
    port = int(os.environ.get('PORT', 80))
    host = '0.0.0.0'  # Explicitly bind to all interfaces
    app = create_app()
    logger.info(f"Starting bot server on port {port}")
    logger.info(f"Access bots at: http://{host}:{port}/<bot_name>")

    # Start server
    serve(app, host=host, port=port, url_scheme='http', threads=4)

def start_bot_process():
    """Start the main bot process"""
    # Import here to avoid circular imports
    from main import main
    import asyncio

    logger.info("Starting bot process")
    try:
        # Run the main function in a separate process
        bot_process = multiprocessing.Process(
            target=lambda: asyncio.run(main()),
            daemon=True
        )
        bot_process.start()
        logger.info(f"Bot process started with PID: {bot_process.pid}")
        return bot_process
    except Exception as e:
        logger.error(f"Failed to start bot process: {e}")
        return None

if __name__ == "__main__":
    # Only start the bot process if this is the main entry point
    # For deployment, we want to run just the server
    port = int(os.environ.get('PORT', 80))
    host = '0.0.0.0'  # Ensure binding to all interfaces
    logger.info(f"Starting server on {host}:{port}")
    app = create_app()
    serve(app, host=host, port=port, url_scheme='http', threads=4)
    
