import json
from flask import Blueprint, request, jsonify

from db_management import get_db_connection
from ai_utils import llm, llm_reasoning, parse_for_answers, get_next_question
from collections import OrderedDict
import os
from flask_jwt_extended import jwt_required
import requests

text_chat_bp = Blueprint('text_chat_bp', __name__)

@text_chat_bp.route('/collect', methods=['POST'])
def collect():
    data = request.get_json()
    form_fields_id = data.get("form_fields_id")
    communication_id = data.get("communication_id")
    last_answer = data.get("answer")
    last_field_id = data.get("field_id")
    last_question = data.get("question")
    reset = (str(data.get("reset", False)).lower() == 'true')
    if form_fields_id is None:
        return jsonify({"error": "Missing form_fields id"}), 400
    communication_context, callback_url, source_id = '', None, None
    form_context = ''
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            if communication_id is not None:
                cursor.execute("""
                    SELECT communication_status
                    FROM communications
                    WHERE communication_id = %s
                """, (communication_id,))
                status_row = cursor.fetchone()
                if status_row and status_row.get("communication_status") == "Completed":
                    return jsonify({"error": "Already submitted"}), 409

            # If no communication_id, create a new communication row.
            if reset and communication_id:
                cursor.execute("""
                    SELECT form_fields, form_context, callback_url
                    FROM form_fields
                    WHERE id = %s AND is_active = 1
                """, (form_fields_id,))
                result = cursor.fetchone()

                if not result:
                    return jsonify({"error": "Invalid form_fields id"}), 400
                form_fields  = json.loads(result["form_fields"])
                form_context = result["form_context"] if result["form_context"] else ''
                # Retrieve the existing communication record.
                sql = "SELECT communication_id, communication_context, source_id FROM communications WHERE communication_id = %s"
                cursor.execute(sql, (communication_id,))
                comm = cursor.fetchone()
                if not comm:
                    return jsonify({"error": "Invalid communication_id"}), 400

                communication_context = comm["communication_context"] if comm["communication_context"] else ''
                callback_url = result["callback_url"] if result["callback_url"] else None
                source_id = comm["source_id"] if comm["source_id"] else None

                


                # Initialize communication state.
                collected_answers = OrderedDict()
                field_asked_counter = {field["field_id"]: 0 for field in form_fields}
                field_parsed_answers = {field["field_id"]: None for field in form_fields}
                language_info = "en-IN"  # default language
                # Update the existing communication row.
                update_sql = """
                    UPDATE communications
                    SET
                        form_fields_id = %s,
                        collected_answers = %s,
                        field_asked_counter = %s,
                        language_info = %s,
                        field_parsed_answers = %s,
                        communication_status = %s
                    WHERE communication_id = %s
                """
                cursor.execute(update_sql, (
                    form_fields_id,
                    json.dumps(collected_answers),
                    json.dumps(field_asked_counter),
                    language_info,
                    json.dumps(field_parsed_answers),
                    'Not Started',
                    communication_id
                ))
                connection.commit()
                start_conversation = True
            elif not communication_id:
                # First, fetch the form_fields from the form_fields table.
                cursor.execute("""
                    SELECT form_fields, form_context
                    FROM form_fields
                    WHERE id = %s AND is_active = 1
                """, (form_fields_id,))
                result = cursor.fetchone()

                if not result:
                    return jsonify({"error": "Invalid form_fields id"}), 400
                form_fields  = json.loads(result["form_fields"])
                form_context = result["form_context"] if result["form_context"] is not None else ''


                # Initialize communication state.
                collected_answers = OrderedDict()
                field_asked_counter = {field["field_id"]: 0 for field in form_fields}
                field_parsed_answers = {field["field_id"]: None for field in form_fields}
                language_info = "en-IN"  # default language

                # Insert the new communication row.
                insert_sql = """
                    INSERT INTO communications 
                    (communication_type, form_fields_id, collected_answers, field_asked_counter, language_info, field_parsed_answers, communication_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_sql, (
                    "text_chat",
                    form_fields_id,
                    json.dumps(collected_answers),
                    json.dumps(field_asked_counter),
                    language_info,
                    json.dumps(field_parsed_answers),
                    'Not Started'
                ))
                connection.commit()
                communication_id = cursor.lastrowid
                start_conversation = True
            else:
                # Retrieve the existing communication record.
                sql = "SELECT communication_id, communication_type, form_fields_id, collected_answers, field_asked_counter, language_info, field_parsed_answers, created_at, updated_at, communication_status, communication_context, source_id FROM communications WHERE communication_id = %s"
                cursor.execute(sql, (communication_id,))
                comm = cursor.fetchone()
                if not comm:
                    return jsonify({"error": "Invalid communication_id"}), 400
                communication_context = comm["communication_context"] if comm["communication_context"] else ''
                
                source_id = comm["source_id"] if comm["source_id"] else None
                # Retrieve the form_fields from the form_fields table using the stored form_fields_id.
                cursor.execute("""
                    SELECT form_fields, form_context, callback_url
                    FROM form_fields
                    WHERE id = %s AND is_active = 1
                """, (form_fields_id,))
                result = cursor.fetchone()
                callback_url = result["callback_url"] if result["callback_url"] else None
                if not result:
                    return jsonify({"error": "Invalid form_fields id"}), 400
                form_fields  = json.loads(result["form_fields"])
                form_context = result["form_context"] if result["form_context"] is not None else ''


                collected_answers = json.loads(comm["collected_answers"], object_pairs_hook=OrderedDict) if comm["collected_answers"] else {}
                field_asked_counter = json.loads(comm["field_asked_counter"]) if comm["field_asked_counter"] else {}
                field_parsed_answers = json.loads(comm["field_parsed_answers"]) if comm["field_parsed_answers"] else {}
                language_info = field_parsed_answers.get('language') if field_parsed_answers.get('language') else "en-IN"
                if not collected_answers and not (last_answer and last_field_id and last_question):
                    start_conversation = True
                else:
                    start_conversation = False
            next_field_id_to_be_asked, additional_context_next_question = None, None
            # If an answer and its field_id are provided, update state.
            if last_answer and last_field_id and last_question:
                # (You can modify the key used to index collected answers as needed.)
                collected_answers[last_question] = last_answer
                field_asked_counter[last_field_id] = field_asked_counter.get(last_field_id, 0) + 1
                field_parsed_answers, next_field_id_to_be_asked, additional_context_next_question = parse_for_answers(
                collected_answers=collected_answers,
                form_fields=form_fields,
                llm=llm_reasoning,
                form_context=form_context,
                field_parsed_answers=field_parsed_answers,
                communication_context=communication_context
                )
            
            # For new conversations, you might want to send a greeting.
            greeting = "Hello! I'm Meera, your AI assistant. It's lovely to connect with you." if start_conversation else None

            # Call get_next_question (assume this function and llm are defined/imported)
            next_field_id, natural_question = get_next_question(
                form_fields=form_fields,
                collected_answers=collected_answers,
                field_parsed_answers=field_parsed_answers,
                field_asked_counter=field_asked_counter,
                llm=llm,
                language=language_info,
                greeting_message=greeting,
                audio=False,
                form_context=form_context,
                next_field_id=next_field_id_to_be_asked,
                additional_context_next_question=additional_context_next_question,
                communication_context=communication_context
            )
            communication_status = 'In Progress'
            if not collected_answers:
                communication_status = 'Not Started'
            # If there is no next field, finish the conversation.
            if next_field_id is None:
                response_data = {
                    "form_fields_id": form_fields_id,
                    "message": natural_question,  # final summary or closing message
                    "field_id": None,
                    "field_parsed_answers": field_parsed_answers,
                    "communication_id": communication_id
                }
                communication_status = 'Completed'
                # Send callback if configured
                if callback_url:
                    try:
                        payload = {
                            'field_parsed_answers': field_parsed_answers,
                            'source_id': source_id
                        }
                        requests.post(callback_url, json=payload, timeout=5)
                    except Exception as exc:
                        # Log but do not interrupt response
                        text_chat_bp.logger.error(f"Callback to {callback_url} failed: {exc}")
            else:
                response_data = {
                    "form_fields_id": form_fields_id,
                    "message": natural_question,
                    "field_id": next_field_id,
                    "field_parsed_answers": field_parsed_answers,
                    "communication_id": communication_id
                }

            # Update the communication record with the new state.
            update_sql = """
                UPDATE communications 
                SET collected_answers = %s, field_asked_counter = %s, field_parsed_answers = %s, language_info = %s, communication_status = %s 
                WHERE communication_id = %s
            """
            cursor.execute(update_sql, (
                json.dumps(collected_answers),
                json.dumps(field_asked_counter),
                json.dumps(field_parsed_answers),
                language_info,
                communication_status,
                communication_id
            ))
            connection.commit()
            # print(collected_answers)
            return jsonify(response_data), 200
    except Exception as e:
        connection.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()



@text_chat_bp.route('/communication/metadata', methods=['PATCH'])
@jwt_required()
def patch_communication_metadata():
    data = request.get_json()
    form_fields_id = data.get("form_fields_id")
    communication_id = data.get("communication_id")
    communication_context = data.get("communication_context")
    source_id = data.get("source_id")
    communication_type = data.get("communication_type") # either text_chat or phone_call
    # Validate required input
    if not form_fields_id:
        return jsonify({"error": "Missing required field: form_fields_id"}), 400
    if not communication_type or communication_type not in ["text_chat", "phone_call"]:
        return jsonify({"error": "Invalid communication_type. Must be either 'text_chat' or 'phone_call'."}), 400
    if not any([communication_context, source_id]):
        return jsonify({"error": "At least one of communication_context or source_id must be provided."}), 400

    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            if communication_id:
                # Check if communication exists
                cursor.execute("SELECT communication_id FROM communications WHERE communication_id = %s", (communication_id,))
                if not cursor.fetchone():
                    return jsonify({"error": "Invalid communication_id"}), 400

                # Update metadata fields
                update_fields = []
                values = []
                if communication_context is not None:
                    update_fields.append("communication_context = %s")
                    values.append(communication_context)
                if source_id is not None:
                    update_fields.append("source_id = %s")
                    values.append(source_id)

                update_sql = f"""
                    UPDATE communications
                    SET {', '.join(update_fields)}
                    WHERE communication_id = %s
                """
                values.append(communication_id)
                cursor.execute(update_sql, tuple(values))

            else:
                # Create empty communication record
                cursor.execute("""
                    SELECT form_fields, form_context
                    FROM form_fields
                    WHERE id = %s AND is_active = 1
                """, (form_fields_id,))
                result = cursor.fetchone()

                if not result:
                    return jsonify({"error": "Invalid form_fields id"}), 400

                form_fields  = json.loads(result["form_fields"])

                collected_answers = OrderedDict()
                field_asked_counter = {field["field_id"]: 0 for field in form_fields}
                field_parsed_answers = {field["field_id"]: None for field in form_fields}
                language_info = "en-IN"

                insert_sql = """
                    INSERT INTO communications 
                    (communication_type, form_fields_id, collected_answers, field_asked_counter, language_info, field_parsed_answers, communication_status,
                     communication_context, source_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(insert_sql, (
                    communication_type,
                    form_fields_id,
                    json.dumps(collected_answers),
                    json.dumps(field_asked_counter),
                    language_info,
                    json.dumps(field_parsed_answers),
                    'Not Started',
                    communication_context,
                    source_id
                ))
                connection.commit()
                communication_id = cursor.lastrowid

        frontend_host = os.getenv("FRONTEND_HOST")
        chat_url = f"https://{frontend_host}/chat?form_fields_id={form_fields_id}&communication_id={communication_id}"
        return jsonify({"url": chat_url}), 200

    except Exception as e:
        connection.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        connection.close()