import logging
logger = logging.getLogger('qurate')
import json
from langchain.schema import SystemMessage, HumanMessage
from datetime import date
import os
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from file_management import cleanup_call_audio_files


# OpenAI API Key

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")


def select_llm(model_name="gpt-4o-mini", provider="open_ai"):
    """Select the language model to use based on the provider."""
    if provider.lower() == "claude":
        return ChatAnthropic(anthropic_api_key=CLAUDE_API_KEY, temperature=0.7, model=model_name)
    else:
        return ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7, model_name=model_name)
llm = select_llm()
llm_reasoning = select_llm(model_name="gpt-4.1-mini", provider="open_ai")
def extract_json(response_text):
    """Extract JSON from LLM response"""
    import re
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error extracting JSON with regex: {e}")
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {}


def parse_for_answers(collected_answers, form_fields, llm, form_context='', field_parsed_answers=None):
    # current_field = last field of field_parsed_answers
    current_field = 'No field answered yet'
    if field_parsed_answers:
        current_field = list(field_parsed_answers.keys())[-1] if field_parsed_answers else 'No field answered yet'
    """Parse collected answers to extract structured data and decide next field to ask"""
    final_prompt = (
        f"form_context: {form_context}. "
        f"Field parsed answers (till now): {field_parsed_answers}\n"
        f"Collected conversation (generated via STT from a phone call; may include disfluencies or transcription errors): {collected_answers}\n"
        f"Field instructions: {form_fields}\n"
        f"Current field: {current_field}\n"
        f"Today's date: {date.today()}. Note: today's date is included for additional context.\n\n"
        "1. Extract each field's value (translated to English, typos corrected) into a JSON object `parsed_fields` "
        "where each key is the field_id and each value is the user's answer or null if unanswered.\n"
        "2. Determine `next_field_id` as follows:\n"
        "   - If `current_field` is incomplete or more data can be deduced from its answer, set `next_field_id` to its field_id.\n"
        "   - Otherwise, from `form_fields`, find the next field whose `condition_for_asking_this_field` is satisfied by `parsed_fields`.\n"
        "   - If a field's `condition_for_asking_this_field` is empty, assume the next field in the order of `form_fields` after the last answered field.\n"
        "   - If no such field exists, set `next_field_id` to null.\n"
        "3. Create `additional_context`: a brief instruction to the AI on how to phrase or frame the next question for `next_field_id`. "
        "If `next_field_id` is null, set `additional_context` to null.\n\n"
        "Return only a JSON object with keys: `parsed_fields`, `next_field_id`, and `additional_context`, and no other text.\n\n"
        "Example return JSON:\n"
        "{"
        "  \"parsed_fields\": {\"language\": \"en-IN\", \"name\": \"Ravi Kumar\", \"email\": null},\n"
        "  \"next_field_id\": \"email\",\n"
        "  \"additional_context\": \"Ask the user for their email address in clear, simple language.\"\n"
        "}"
    ).strip()

    messages = [
        SystemMessage(content="You are an assistant that extracts structured information from text and plans the next question."),
        HumanMessage(content=final_prompt)
    ]
    try:
        json_llm = llm
        final_output = json_llm.invoke(messages).content.strip()
        result = extract_json(final_output)
        print(f"\n[Debug] Parsed result: {json.dumps(result, indent=2)}")
        return result.get("parsed_fields", {}), result.get("next_field_id"), result.get("additional_context")
    except Exception as e:
        logger.error(f"Error parsing answers: {e}")
        return {}, None, None


def generate_summary_response(field_parsed_answers, form_fields, llm, language="en-IN"):
    """
    Generate a natural language summary of all collected data before ending the call.
    
    Args:
        field_parsed_answers (dict): Dictionary of field IDs to their values
        form_fields (list): List of field definitions
        llm: The language model to use for generation
        language (str): The language code to generate the response in
    
    Returns:
        str: A natural language summary of the collected data
    """
    # Create a mapping from field_id to field_name for better readability
    field_names = {field["field_id"]: field["field_name"] for field in form_fields}
    
    # Build a readable representation of the collected data
    collected_data_str = ""
    for field_id, value in field_parsed_answers.items():
        if value is not None and value != "":
            field_name = field_names.get(field_id, field_id)
            collected_data_str += f"{field_name}: {value}\n"
    
    # If no data was collected, return a simple thank you
    if not collected_data_str.strip():
        return "Thank you for your time."
    
    language_prompt = language if language != None and not language.lower().startswith('en') else "English"
    
    prompt = f"""
    You are a helpful assistant summarizing information collected during a phone call.
    Today's date: {date.today()}.
    Please create a friendly summary of the following collected information in {language_prompt}:
    
    {collected_data_str}
    
    The summary should:
    1. Start with "I have collected the following information:"
    2. List each piece of information in a natural, conversational way
    3. End with a warm thank you for the person's time
    4. Be polite and professional
    5. Be concise (no more than 5 sentences total)
    
    The entire summary should sound natural when spoken over the phone with no extra commentary..
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant that summarizes information in a natural, conversational way."),
        HumanMessage(content=prompt)
    ]
    
    try:
        summary = llm.invoke(messages).content.strip()
        if ":" in summary:
            summary = summary.split(":", 1)[1].strip()
        else:
            summary = summary
        return summary
    except Exception as e:
        logger.error(f"Error generating summary response: {e}")
        # Fallback to a simple formatted response if the LLM fails
        summary = "I have collected the following information:\n"
        for field_id, value in field_parsed_answers.items():
            if value is not None and value != "":
                field_name = field_names.get(field_id, field_id)
                summary += f"{field_name}: {value}. "
        summary += "\nThank you for your time."
        return summary




def get_next_question(form_fields, collected_answers, field_parsed_answers, field_asked_counter, llm, language="en-GB", 
                      greeting_message=None, call_id=None, audio = True, form_context='', next_field_id=None, 
                      additional_context_next_question=None):
    """Generate the next question based on collected answers and question attempts"""
    if next_field_id:
        # If a specific field is requested, find it in the form fields
        pending_fields = [
            field for field in form_fields 
            if field["field_id"] == next_field_id
        ]
    if not next_field_id or not pending_fields:    
        pending_fields = [
            field for field in form_fields 
            if field_asked_counter.get(field["field_id"], 0) < 3 and 
            (field_parsed_answers.get(field["field_id"]) is None or field_parsed_answers.get(field["field_id"]) == "")
        ]
    

    language_prompt = language if language != None and not language.lower().startswith('en') else "English"


    if audio and not pending_fields:
        # Modified to just return a simple thank-you without asking for more
        cleanup_call_audio_files(call_id)
        return None, generate_summary_response(field_parsed_answers, form_fields, llm, language=language_prompt)
    
    if not pending_fields:
        return None, "Thank you for your time."
    
    next_field = pending_fields[0]
    # Use provided language if available; otherwise, default to "English"
    
    last_n_conversations = list(collected_answers.items())[-20:]
    context = "\n".join([f"System: {quest} -> User Response: {ans}" for quest, ans in last_n_conversations])
    current_attempt = field_asked_counter.get(next_field["field_id"], 0) + 1
    feedback_instruction = ""
    if current_attempt > 1 and additional_context_next_question:
        feedback_instruction = f'Instructions for asking this question: {additional_context_next_question}. '
    question_prompt = (
        f'Context: {form_context}. '
        f'Please generate a relevant and engaging question in {language_prompt} '
        f'(e.g., en-IN for English (Indian Accent), hi-IN for Hindi). '
        f'This question should help collect the field data: {next_field["field_name"]}. '
        f'{feedback_instruction}'
        f'Background: {next_field["additional_info"]}. '
        f'Use simple, conversational phrasing—avoid formal or dictionary-style language. '
        f'If the target language is a local Indian language (like hi-IN, mr-IN, ta-IN), '
        f'weave in common English words or short phrases to improve clarity. '
        f'Provide only the question itself, with no extra commentary. '
        f'This is attempt {current_attempt} for this field. '
        f'If this is not the first attempt, try a different approach. '
        f'Here is our recent conversation context: {context or "No previous context"}. '
        f'Remember that we have already collected some answers: {field_parsed_answers}. '
        f"Today's date: {date.today()}. Note: today's date is included for additional context."
        'Feel free to ask follow-up questions or seek clarification if previous responses were unclear. '
        'Tone: Show compassion and warmth in your question.'
    ).strip()
    # print(f"\n[Debug] Question prompt: {question_prompt}")
 
    messages = [
        SystemMessage(content="You are a conversational human that frames questions naturally. "
        "As a subject matter expert, your role is to thoughtfully guide the process of collecting meaningful and respectful information. "
        "Please generate a relevant, engaging, and empathetic question that encourages honest sharing and provides deep insight into the topic, "
        "while being sensitive to the context and experiences of the people involved."),
        HumanMessage(content=question_prompt)
    ]
    try:
        natural_question = llm.invoke(messages).content.strip()
        if ":" in natural_question:
            natural_question = natural_question.split(":", 1)[1].strip().replace("`", "").replace('"', "")
        else:
            natural_question = natural_question.strip().replace("`", "").replace('"', "")
        if greeting_message and not collected_answers:
            beep_message = ""
            # if audio:
            #     beep_message = "I'll ask you a few quick questions. After each one, you can share your answer. Once you’ve finished speaking, you’ll hear a sound—that means your response has been recorded. Let’s start with this:"

            natural_question = f"{greeting_message}. {beep_message} {natural_question}"
            natural_question = natural_question.strip()

        return next_field["field_id"], natural_question
    except Exception as e:
        logger.error(f"LLM error in get_next_question: {e}")
        return next_field["field_id"], f"What is your {next_field['field_name']}?"



def parse_for_form_fields(user_query, llm, form_context=''):
    """
    Parse user query to generate form fields using LLM.
    Returns a dictionary with a 'fields' key containing the field definitions.
    Each field will include validation rules, conditional display logic, and parsing comments.
    """
    human_message = f"""
        You are a Form Design Specialist at expert level in collecting and structuring data.
        As an SME, you always know which fields are required—even if the user doesn't mention them.
        Given the user's description of what they want to collect, provide a JSON object with:
        - "field_id": A unique identifier for the field.
        - "field_name": The display name of the field.
        - "datatype": The type of data expected ("<string|number|date|email|phone|boolean>").
        - "additional_info": Extra details about what the field is used for.
        - "validation": A free-form text description of the validation rules for the field.
        - "additional_parsing_comments": Any extra notes on how to parse or interpret this field.
        - "condition_for_asking_this_field": A conditional expression describing when to ask this field (e.g., 'ask if name was answered').

        Return a JSON object following this exact structure:
        {{
            "fields": [
                {{
                    "field_id": "language",
                    "field_name": "Language",
                    "datatype": "string",
                    "additional_info": "This question is asked so that further communication can happen in that language",
                    "validation": "Input must be the name of an Indian language (including English); anything else is invalid",
                    "additional_parsing_comments": "Map the user's input (e.g. English, Hindi, Marathi) to its BCP 47 tag (en-IN, hi-IN, mr-IN). Reject non-Indian languages (e.g. Spanish).",
                    "condition_for_asking_this_field": null
                }},
                {{
                    "field_id": "<unique_field_id>",
                    "field_name": "<Human-readable Label>",
                    "datatype": "<string|number|date|email|phone|boolean>",
                    "additional_info": "<why it's needed or any notes like validation rules>",
                    "validation": "<rule description>",
                    "additional_parsing_comments": "<any parser-specific notes>",
                    "condition_for_asking_this_field": "ask if <other_field_id> was answered"
                }}
            ]
        }}

        **Guidance for ordering**  
        1. **language** (always first—to establish preferred communication)  
        2. **identity** (e.g. name, username)  
        3. **contact** (email, phone, address, etc.)  
        4. **topic-specific** fields (in the logical sequence an interviewer would ask)  
        5. **closing** or **comments** (anything else needed to complete or wrap up)

        Requirements:
        1. Always include a “language” field first, with:
           - field_id: "language"
           - field_name: "Language"
           - datatype: "string"
           - additional_info: "So we can communicate in the user's preferred language."
           - validation: "Input must be the name of an Indian language (including English); anything else is invalid"
           - additional_parsing_comments: "Map the user's input (e.g. English, Hindi, Marathi) to its BCP 47 tag (en-IN, hi-IN, mr-IN). Reject non-Indian languages (e.g. Spanish)."
           - condition_for_asking_this_field: ""
        2. Infer and include any other logical contact and identity fields (e.g. name, email, phone, address, date_of_birth), including appropriate validation and display logic.
        3. For each domain need (survey questions, appointment details, feedback, etc.), add sensible fields in a logical order and include validation, parsing comments, and conditional logic as needed.
        4. Do not output anything outside the JSON.

        User Query: {user_query}
        Form Context: {form_context}
        Return only valid JSON.
    """

    messages = [
        SystemMessage(content="You are an assistant that extracts structured information from text."),
        HumanMessage(content=human_message)
    ]
    try:
        # Attempt to bind JSON response format if supported
        json_llm = llm
        if hasattr(llm, 'bind') and callable(getattr(llm, 'bind')):
            json_llm = llm.bind(response_format={"type": "json_object"})
    except Exception as e:
        logger.warning(f"Could not set response_format for LLM: {e}")

    try:
        final_output = json_llm.invoke(messages).content.strip()
        form_fields = extract_json(final_output).get("fields", [])
        # Ensure language field is first with our strict rules
        if not form_fields or form_fields[0].get("field_id") != "language":
            form_fields.insert(0, {
                "field_id": "language",
                "field_name": "Language",
                "datatype": "string",
                "additional_info": "This question is asked so that further communication can happen in that language",
                "validation": "Input must be the name of an Indian language (including English); anything else is invalid",
                "additional_parsing_comments": "Map the user's input (e.g. English, Hindi, Marathi) to its BCP 47 tag (en-IN, hi-IN, mr-IN). Reject non-Indian languages (e.g. Spanish).",
                "condition_for_asking_this_field": ''
            })
        return {"fields": form_fields}
    except Exception as e:
        logger.error(f"Error in parse_for_form_fields: {e}")
        return {"fields": []}

def get_default_form_fields():
    """Return default form fields when query parsing fails or is not provided"""
    return [
        {
            "field_id": "language",
            "field_name": "Language",
            "datatype": "string",
            "additional_info": "This question is asked so that further communication can happen in that language",
            "validation": "Input must be the name of an Indian language (including English); anything else is invalid",
            "additional_parsing_comments": "Map the user's input (e.g. English, Hindi, Marathi) to its BCP 47 tag (en-IN, hi-IN, mr-IN). Reject non-Indian languages (e.g. Spanish).",
            "condition_for_asking_this_field": None
        },
        {
            "field_id": "name",
            "field_name": "Name",
            "datatype": "string",
            "additional_info": "User's full name",
            "validation": "Must be a non-empty string",
            "additional_parsing_comments": "",
            "condition_for_asking_this_field": None
        }
    ]



