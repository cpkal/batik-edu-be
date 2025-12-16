from flask import Flask
from routes import api
from services.batik.rag import RAGService
import os
from dotenv import load_dotenv
from google import genai
from containers import Container
from errors import register_error_handlers
from flask_cors import CORS

load_dotenv()

def create_app():
	app = Flask(__name__)

	# setup this later
	CORS(app)

	register_error_handlers(app)

	container = Container()
	app.container = container

	app.register_blueprint(api, url_prefix='/api/v1')

	return app

if __name__ == '__main__':
	app = create_app()
	app.run(host='0.0.0.0', port=8000)