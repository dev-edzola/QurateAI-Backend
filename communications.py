from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from db_management import get_db_connection
from common_utils import _safe_json_load, _format_datetime

communications_bp = Blueprint('communications_bp', __name__)

@communications_bp.route('/communications', methods=['GET'])
@jwt_required()
def list_communications():
    current_user_id   = get_jwt_identity()
    form_fields_id    = request.args.get('form_fields_id', type=int, default=None)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Base query
            sql = """
                SELECT
                  c.communication_type,
                  c.communication_id,
                  c.form_fields_id,
                  c.collected_answers,
                  c.field_parsed_answers,
                  c.updated_at,
                  c.communication_status
                FROM communications AS c
                LEFT JOIN form_fields   AS f
                  ON c.form_fields_id = f.id AND f.user_id IS NOT NULL
                WHERE f.user_id = %s AND f.user_id IS NOT NULL AND c.communication_status != 'Not Started'
            """
            params = [current_user_id]

            # If a specific form_fields_id was provided, add it to the WHERE clause
            if form_fields_id is not None:
                sql += " AND c.form_fields_id = %s AND c.form_fields_id IS NOT NULL"
                params.append(form_fields_id)

            # Order by newest first
            sql += " ORDER BY c.updated_at DESC"

            cursor.execute(sql, params)
            rows = cursor.fetchall()

        # Normalize to JSON‑serializable Python types
        communications = []
        for r in rows:
            communications.append({
                "communication_id":     r["communication_id"],
                "communication_type":   r["communication_type"],
                "form_fields_id":       r["form_fields_id"],
                "collected_answers":    _safe_json_load(r["collected_answers"]),
                "field_parsed_answers": _safe_json_load(r["field_parsed_answers"]),
                "updated_at":           _format_datetime(r["updated_at"]),
                "communication_status": r["communication_status"]
            })

        return jsonify({"communications": communications}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        conn.close()
   