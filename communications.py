from flask import Blueprint, jsonify, request, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
from db_management import get_db_connection
from common_utils import _safe_json_load, _format_datetime
import io
import pandas as pd


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
   

@communications_bp.route('/communications/export/<int:form_fields_id>', methods=['GET'])
@jwt_required()
def export_communications_csv(form_fields_id):
    user_id = get_jwt_identity()
    conn = get_db_connection()
    try:
        # 1) Fetch everything we need
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT
                  c.communication_id,
                  c.communication_type,
                  f.form_field_name,
                  c.field_parsed_answers,
                  c.updated_at,
                  c.communication_status
                FROM communications AS c
                JOIN form_fields AS f
                  ON c.form_fields_id = f.id
                WHERE f.user_id = %s AND f.user_id IS NOT NULL AND c.communication_status != 'Not Started'
                 AND c.form_fields_id = %s AND c.form_fields_id IS NOT NULL
                ORDER BY c.updated_at DESC
            """, (user_id,form_fields_id))
            rows = cursor.fetchall()

        # 2) Build list of dicts, collect all JSON keys
        records = []
        all_keys = set()
        for r in rows:
            parsed = _safe_json_load(r['field_parsed_answers'])
            all_keys.update(parsed.keys())
            rec = {
                'communication_id':     r['communication_id'],
                'communication_type':   r['communication_type'],
                'form_field_name':      r['form_field_name'],
                **parsed,
                'updated_at':           _format_datetime(r['updated_at']),
                'communication_status': r['communication_status']
            }
            records.append(rec)

        # 3) Build DataFrame, ensure consistent column ordering
        df = pd.DataFrame(records)
        cols = [
            'communication_id',
            'communication_type',
            'form_field_name',
            *sorted(all_keys),
            'updated_at',
            'communication_status'
        ]
        df = df.reindex(columns=cols).fillna('')
        # 4) Stream out as CSV
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return Response(
            buf.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=communications.csv'}
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        conn.close()