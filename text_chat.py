import json
from flask import Blueprint, request, jsonify

from db_management import get_db_connection
from ai_utils import llm, llm_reasoning, parse_for_answers, get_next_question
from collections import OrderedDict


text_chat_bp = Blueprint('text_chat_bp', __name__)

@text_chat_bp.route('/collect', methods=['POST'])
def collect():
    data = request.get_json()
    form_fields_id = data.get("form_fields_id")
    communication_id = data.get("communication_id")
    last_answer = data.get("answer")
    last_field_id = data.get("field_id")
    last_question = data.get("question")
    
    if form_fields_id is None:
        return jsonify({"error": "Missing form_fields id"}), 400
    
    form_context = ''
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            # If no communication_id, create a new communication row.
            if not communication_id:
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
                sql = "SELECT * FROM communications WHERE communication_id = %s"
                cursor.execute(sql, (communication_id,))
                comm = cursor.fetchone()
                if not comm:
                    return jsonify({"error": "Invalid communication_id"}), 400

                # Retrieve the form_fields from the form_fields table using the stored form_fields_id.
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


                collected_answers = json.loads(comm["collected_answers"], object_pairs_hook=OrderedDict) if comm["collected_answers"] else {}
                field_asked_counter = json.loads(comm["field_asked_counter"]) if comm["field_asked_counter"] else {}
                field_parsed_answers = json.loads(comm["field_parsed_answers"]) if comm["field_parsed_answers"] else {}
                language_info = field_parsed_answers.get('language') if field_parsed_answers.get('language') else "en-IN"
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
                field_parsed_answers=field_parsed_answers
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
                additional_context_next_question=additional_context_next_question
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
