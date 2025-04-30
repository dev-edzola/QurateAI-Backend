import os
import json
from flask import Blueprint, request, jsonify

from db_management import get_db_connection
from flask_jwt_extended import (
    jwt_required,
    get_jwt_identity,
)
from ai_utils import select_llm, parse_for_form_fields

llm = select_llm(model_name="gpt-4.1-mini", provider="open_ai")
frontend_host = os.environ.get('FRONTEND_HOST')

form_bp = Blueprint('form_bp', __name__)

@form_bp.route('/generate_form_fields', methods=['POST'])
@jwt_required()
def generate_form_fields():
    current_user_id = get_jwt_identity()
    data = request.get_json()
    user_query = data.get("user_query")
    form_field_name = data.get("form_field_name")
    form_context = data.get("form_context")
    if not user_query:
        return jsonify({"error": "Missing user_query"}), 400
    if not form_field_name:
        return jsonify({"error": "Missing form_field_name"}), 400

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # 1) Check for an existing form_fields entry
            check_sql = """
                SELECT id
                  FROM QURATE_AI.form_fields
                 WHERE user_id = %s
                   AND form_field_name = %s
                LIMIT 1
            """
            cursor.execute(check_sql, (current_user_id, form_field_name))
            existing = cursor.fetchone()

            if existing:
                # 2) If found, return a conflict response
                return jsonify({
                    "error": "Form with this name already exists",
                    "form_fields_id": existing["id"]
                }), 409

            # 3) Otherwise insert new
            # Generate the fields via your LLM helper
            form_fields = parse_for_form_fields(user_query, llm).get("fields", [])
            if not form_fields:
                return jsonify({"error": "No form fields generated"}), 400
            insert_sql = """
                INSERT INTO QURATE_AI.form_fields
                    (user_id, form_field_name, form_context, form_fields)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_sql,
                           (current_user_id,
                            form_field_name,
                            form_context,
                            json.dumps(form_fields)))
            connection.commit()
            new_id = cursor.lastrowid

        form_link = f"https://{frontend_host}/chat?form_fields_id={new_id}"
        return jsonify({
            "form_link": form_link,
            "form_fields_id": new_id,
            "form_fields": form_fields
        }), 201

    except Exception as e:
        connection.rollback()
        return jsonify({"error": str(e)}), 500

    finally:
        connection.close()


@form_bp.route('/forms', methods=['GET'])
@jwt_required()
def get_active_forms():
    current_user_id = get_jwt_identity()
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = (
                "SELECT id, form_field_name, form_fields, is_active, created_at, updated_at FROM QURATE_AI.form_fields "
                "WHERE user_id IS NOT NULL AND user_id = %s AND is_active = 1 "
                "ORDER BY updated_at DESC"
            )
            cursor.execute(sql, (current_user_id,))
            forms = cursor.fetchall()
        return jsonify(forms), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()

@form_bp.route('/all_forms', methods=['GET'])
@jwt_required()
def get_all_forms():
    current_user_id = get_jwt_identity()
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = (
                "SELECT id, form_field_name, form_fields, is_active, created_at, updated_at, form_context FROM QURATE_AI.form_fields "
                "WHERE user_id IS NOT NULL AND user_id = %s"
                "ORDER BY updated_at DESC"
            )
            cursor.execute(sql, (current_user_id,))
            forms = cursor.fetchall()
        return jsonify(forms), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()        


@form_bp.route('/forms/<int:form_id>', methods=['GET'])
@jwt_required()
def get_specific_form(form_id):
    current_user_id = get_jwt_identity()
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = (
                "SELECT id, form_field_name, form_fields, is_active, created_at, updated_at, form_context FROM QURATE_AI.form_fields "
                "WHERE id = %s AND user_id IS NOT NULL AND user_id = %s"
            )
            cursor.execute(sql, (form_id, current_user_id))
            form = cursor.fetchone()
            if not form:
                return jsonify({"error": "Form not found or inactive"}), 404
        return jsonify(form), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()


@form_bp.route('/forms/<int:form_id>', methods=['PATCH'])
@jwt_required()
def update_form(form_id):
    current_user_id = get_jwt_identity()
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    update_fields = []
    values = []
    if "form_field_name" in data:
        update_fields.append("form_field_name = %s")
        values.append(data["form_field_name"])
    if "form_fields" in data:
        update_fields.append("form_fields = %s")
        values.append(json.dumps(data["form_fields"]))
    if "is_active" in data:
        update_fields.append("is_active = %s")
        values.append(data["is_active"])
    if "form_context" in data:
        update_fields.append("form_context = %s")
        values.append(json.dumps(data["form_context"]))    

    if not update_fields:
        return jsonify({"error": "No valid fields provided for update"}), 400

    update_clause = ", ".join(update_fields)
    sql = (
        f"UPDATE QURATE_AI.form_fields SET {update_clause} "
        "WHERE id = %s AND user_id IS NOT NULL AND user_id = %s"
    )
    values.extend([form_id, current_user_id])

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, tuple(values))
            if cursor.rowcount == 0:
                return jsonify({"error": "Form not found or no permission."}), 404
            connection.commit()
        return jsonify({"message": "Form updated successfully."}), 200
    except Exception as e:
        connection.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()


@form_bp.route('/forms/<int:form_id>', methods=['DELETE'])
@jwt_required()
def delete_form(form_id):
    current_user_id = get_jwt_identity()
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            sql = (
                "UPDATE QURATE_AI.form_fields SET is_active = 0 "
                "WHERE id = %s AND user_id IS NOT NULL AND user_id = %s"
            )
            cursor.execute(sql, (form_id, current_user_id))
            if cursor.rowcount == 0:
                return jsonify({"error": "Form not found or no permission."}), 404
            connection.commit()
        return jsonify({"message": "Form marked as inactive."}), 200
    except Exception as e:
        connection.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()