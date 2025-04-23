import os
from dotenv import load_dotenv

load_dotenv()

env = os.environ.get("ENVIRONMENT", "production").lower()

if env == "development":
    load_dotenv(override=True)


import eventlet
eventlet.monkey_patch()

from flask import Flask
from extensions import sock
from flask_cors import CORS
import file_management
file_management.init_file_management()

from auth import auth_bp
from forms import form_bp
from text_chat import text_chat_bp
from phone_call_twilio import phone_call_twilio_bp
from communications import communications_bp
import datetime
from auth import configure_jwt_callbacks
from flask_jwt_extended import JWTManager
from mail_config import mail, init_mail

# Initialize Flask app
app = Flask(__name__)

#–– Mail config ––
init_mail(app)



app.config["JWT_SECRET_KEY"] = os.environ["JWT_SECRET_KEY"]
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = datetime.timedelta(hours=1)
app.config['ACTIVATION_SALT']     = os.environ.get('ACTIVATION_SALT')
app.config["PASSWORD_RESET_SALT"] = os.environ["PASSWORD_RESET_SALT"]


jwt = JWTManager(app)
configure_jwt_callbacks(jwt)
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(form_bp) 
app.register_blueprint(text_chat_bp)
app.register_blueprint(phone_call_twilio_bp) 
app.register_blueprint(communications_bp)

app.config["DEBUG"] = False
app.config["ENV"] = "production"
app.config["SECRET_KEY"] = os.urandom(24)  # Keep this as is for now

sock.init_app(app)

# Enable CORS for the Flask app
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:8080",           # Local development
            "https://qurate-ai-frontend.onrender.com",  # Production frontend
            "https://twilio-flask-ysez.onrender.com"
        ],
        "methods": ["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type","Authorization"]
    }
})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)


