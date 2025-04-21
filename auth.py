import datetime
import jwt as pyjwt
from flask import Blueprint, request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash
from db_management import get_db_connection
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    get_jwt
)

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

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify(message="Missing required fields."), 400

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM users WHERE username=%s OR email=%s", (username, email)
            )
            if cursor.fetchone():
                return jsonify(message="User already exists."), 409

            pwd_hash = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, pwd_hash)
            )
            conn.commit()
        return jsonify(message="User created."), 201
    except Exception as e:
        conn.rollback()
        return jsonify(error=str(e)), 500
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
                "SELECT id, password_hash FROM users WHERE username=%s", (username,)
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

        if not user:
            return jsonify(message="No user with that email."), 404

        reset_token = pyjwt.encode(
            {
                'user_id': user['id'],
                'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
            },
            current_app.config['JWT_SECRET_KEY'],
            algorithm="HS256"
        )
        return jsonify(reset_token=reset_token), 200
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        conn.close()

@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json() or {}
    token = data.get('reset_token')
    new_pwd = data.get('new_password')

    if not token or not new_pwd:
        return jsonify(message="Token and new password required."), 400

    try:
        decoded = pyjwt.decode(
            token,
            current_app.config['JWT_SECRET_KEY'],
            algorithms=["HS256"]
        )
        user_id = decoded['user_id']
    except pyjwt.ExpiredSignatureError:
        return jsonify(message="Reset token expired."), 401
    except pyjwt.InvalidTokenError:
        return jsonify(message="Invalid reset token."), 401

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
