import datetime
import jwt as pyjwt
from flask import Blueprint, request, jsonify, current_app, render_template
from werkzeug.security import generate_password_hash, check_password_hash
from db_management import get_db_connection
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    get_jwt
)
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from mail_config import mail
from flask_mail import Message

auth_bp = Blueprint('auth_bp', __name__)


def configure_jwt_callbacks(jwt):
    @jwt.token_in_blocklist_loader
    def check_if_revoked(jwt_header, jwt_payload):
        jti = jwt_payload["jti"]
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM blacklisted_tokens WHERE jti=%s", (jti,))
                return cur.fetchone() is not None
        finally:
            conn.close()

# Serializer helper
def get_serializer():
    return URLSafeTimedSerializer(current_app.config['JWT_SECRET_KEY'])

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    username = data.get('username')
    email    = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify(message="Missing required fields."), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Check existing user
            cursor.execute(
                "SELECT id FROM users WHERE username=%s OR email=%s",
                (username, email)
            )
            if cursor.fetchone():
                return jsonify(message="User already exists."), 409

            # Insert inactive user
            pwd_hash = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, Active) "
                "VALUES (%s, %s, %s, 0)",
                (username, email, pwd_hash)
            )
            user_id = cursor.lastrowid
            conn.commit()

        # Generate activation token (valid 24h)
        serializer = get_serializer()
        token = serializer.dumps(
            user_id,
            salt=current_app.config['ACTIVATION_SALT']
        )
        frontend_host = current_app.config['FRONTEND_HOST']
        activate_url = (
            f"https://{frontend_host}/activate"
            f"?token={token}"
        )

        # Send email
        msg = Message(
            subject="Activate Your Qurate-AI Account",
            recipients=[email]
        )
        msg.html = render_template(
            'activate_account_email.html',
            username=username,
            activate_url=activate_url
        )
        mail.send(msg)

        return jsonify(message="User created. Activation email sent."), 201

    except Exception as e:
        conn.rollback()
        current_app.logger.exception("Error in signup")
        return jsonify(error="Signup failed."), 500

    finally:
        conn.close()


@auth_bp.route('/activate', methods=['GET'])
def activate_account():
    token = request.args.get('token')
    if not token:
        return jsonify(message="Activation token required."), 400

    serializer = get_serializer()
    try:
        user_id = serializer.loads(
            token,
            salt=current_app.config['ACTIVATION_SALT'],
            max_age=24 * 3600  # 24 hours
        )
    except SignatureExpired:
        return jsonify(message="Activation link expired."), 400
    except BadSignature:
        return jsonify(message="Invalid activation token."), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET Active = 1 WHERE id = %s",
                (user_id,)
            )
            conn.commit()
        return jsonify(
            message="Account activated successfully.",
            activated=True
        ), 200

    except Exception:
        current_app.logger.exception("Error activating account")
        return jsonify(error="Account activation failed."), 500

    finally:
        conn.close()

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify(message="Missing credentials."), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, password_hash FROM users WHERE username=%s AND Active = 1", (username,)
            )
            user = cursor.fetchone()

        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify(message="Invalid credentials."), 401

        access_token = create_access_token(identity=str(user['id']))
        refresh_token = create_refresh_token(identity=str(user['id']))
        return jsonify(access_token=access_token, refresh_token=refresh_token), 200
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        conn.close()

@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    user_id = get_jwt_identity()
    new_access = create_access_token(identity=user_id)
    return jsonify(access_token=new_access), 200

@auth_bp.route('/forget-password', methods=['POST'])
def forget_password():
    data = request.get_json() or {}
    email = data.get('email')
    if not email:
        return jsonify(message="Email required."), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
            user = cursor.fetchone()

        # Always respond 200 to avoid revealing whether the email exists
        if user:
            serializer = get_serializer()
            token = serializer.dumps(
                user['id'],
                salt=current_app.config['PASSWORD_RESET_SALT']
            )
            frontend_host = current_app.config['FRONTEND_HOST']
            reset_url = (
                f"https://{frontend_host}"
                f"/reset-password?reset_id={token}"
            )

            msg = Message(
                subject="Reset Your Qurate-AI Password",
                recipients=[email]
            )
            msg.html = render_template(
                'reset_password_email.html',
                reset_url=reset_url
            )
            mail.send(msg)

        return jsonify(message="If that email exists, a reset link has been sent."), 200

    except Exception:
        current_app.logger.exception("Error in forget-password")
        return jsonify(error="Unable to send reset email."), 500

    finally:
        conn.close()

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json() or {}
    token   = data.get('reset_token')
    new_pwd = data.get('new_password')

    if not token or not new_pwd:
        return jsonify(message="Token and new password required."), 400

    # 1. Load & verify the token
    serializer = get_serializer()
    try:
        user_id = serializer.loads(
            token,
            salt=current_app.config['PASSWORD_RESET_SALT'],
            max_age=15 * 60  # seconds
        )
    except SignatureExpired:
        return jsonify(message="Reset token expired."), 401
    except BadSignature:
        return jsonify(message="Invalid reset token."), 401

    # 2. Update the password
    conn = get_db_connection()
    try:
        pwd_hash = generate_password_hash(new_pwd)
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET password_hash=%s WHERE id=%s",
                (pwd_hash, user_id)
            )
            conn.commit()
        return jsonify(message="Password reset successful."), 200
    except Exception as e:
        conn.rollback()
        return jsonify(error=str(e)), 500
    finally:
        conn.close()

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    exp_ts = get_jwt()["exp"]
    expires_at = datetime.datetime.fromtimestamp(exp_ts)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO blacklisted_tokens (jti, expires_at) VALUES (%s, %s)",
                (jti, expires_at)
            )
            conn.commit()
        return jsonify(message="Logged out successfully."), 200
    except Exception as e:
        conn.rollback()
        return jsonify(error=str(e)), 500
    finally:
        conn.close()
