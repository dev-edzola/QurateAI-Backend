# mail_config.py

import os
from flask_mail import Mail

mail = Mail()

def init_mail(app):
    """Configure Flask-Mail on the given app."""
    app.config.update(
        MAIL_SERVER         = os.environ.get("MAIL_SERVER"),
        MAIL_PORT           = int(os.environ.get("MAIL_PORT", 587)),
        MAIL_USE_TLS        = os.environ.get("MAIL_USE_TLS", "True") == "True",
        MAIL_USERNAME       = os.environ.get("MAIL_USERNAME"),
        MAIL_PASSWORD       = os.environ.get("MAIL_PASSWORD"),
        MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER"),
    )
    mail.init_app(app)
