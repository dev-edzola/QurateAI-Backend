from json import loads, dumps, JSONDecodeError
import os
from dotenv import load_dotenv
from sys import exit
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from re import search, DOTALL
from openai import OpenAI
from tempfile import NamedTemporaryFile
import time
import numpy as np
#import sounddevice as sd
import soundfile as sf
import requests
import base64
from pydantic import BaseModel
from google.cloud import speech
import io
import datetime
# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
TTS_URL = "https://api.sarvam.ai/text-to-speech"      
STT_URL = "https://api.sarvam.ai/speech-to-text"

# =============================================================================
def extract_json(response_text):
    # This regex finds content within triple backticks that starts with "json"
    match = search(r"```json\s*(\{.*?\})\s*```", response_text, DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return loads(json_str)
        except JSONDecodeError:
            pass
    # Fallback: try to parse the entire response as JSON
    try:
        return loads(response_text)
    except JSONDecodeError:
        return {}

# =============================================================================
def select_llm(choice, model_name) -> object:
    """    
    Parameters:
      - choice (str): LLM Name
      - model_name (str): Name of model
      
    Returns:
      - llm (Object)
    """
    if choice == "open_ai":
        api_key = OPENAI_API_KEY
        if not api_key:
            print("Error: OPENAI_API_KEY not set.")
            exit(1)
        return ChatOpenAI(openai_api_key=api_key, temperature=0.7, model_name=model_name)
    else:
        return ChatOpenAI(openai_api_key=api_key, temperature=0.7, model_name="gpt-4o-mini")

# =============================================================================

def open_ai_tts(question, voice = "alloy", model = "tts-1-hd"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Create the speech audio via OpenAI TTS
    with NamedTemporaryFile(suffix=".mp3", delete=False) as tts_temp:
        tts_file = tts_temp.name
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=question,
    ) as tts_response:
        tts_response.stream_to_file(tts_file)
    # Play the audio (blocking call)
    #playsound(tts_file)
    os.remove(tts_file)

def sarvam_ai_tts(question, model = "bulbul:v1"):
    tts_payload = {
        "inputs": [question],
        "target_language_code": "en-IN",
        "speaker": "meera",             
        "pitch": 0,
        "pace": 1.2,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": model,
        "eng_interpolation_wt": 123,
        "override_triplets": {}
    }
    #print(tts_payload)
    tts_headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    tts_response = requests.post(TTS_URL, json=tts_payload, headers=tts_headers)
    if tts_response.status_code != 200:
        print("Error with TTS:", tts_response.text)
        return None
    tts_json = tts_response.json()
    if "audios" not in tts_json or not tts_json["audios"]:
        print("No audio received from TTS.")
        return None
    # Decode the first base64-encoded WAV audio
    audio_base64 = tts_json["audios"][0]
    audio_bytes = base64.b64decode(audio_base64)
    with NamedTemporaryFile(suffix=".wav", delete=False) as tts_temp:
        tts_file = tts_temp.name
    with open(tts_file, "wb") as f:
        f.write(audio_bytes)
    # Play the generated speech
    #playsound(tts_file)
    os.remove(tts_file)

def sarvam_ai_stt(language_code, start_conversation):
    silence_duration_required = 3  # Seconds of silence to stop recording
    print(f"Listening for your response (recording will stop after {silence_duration_required} sec of silence)...")
    fs = 44100           # Sample rate
    channels = 1         # Mono recording
    chunk_duration = 0.5 # Seconds per chunk
    silence_threshold = 0.01  # RMS amplitude threshold for silence; adjust as needed
    
    recorded_chunks = []
    silence_time = 0

    while True:
        chunk = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=channels, dtype='float32')
        sd.wait()
        recorded_chunks.append(chunk)
        rms = np.sqrt(np.mean(chunk**2))
        if rms < silence_threshold:
            silence_time += chunk_duration
        else:
            silence_time = 0
        if silence_time >= silence_duration_required:
            print("Silence detected. Stopping recording.")
            break

    audio_data = np.concatenate(recorded_chunks, axis=0)
    # Save the recorded audio to a temporary WAV file
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_filename = temp_wav.name
    sf.write(temp_filename, audio_data, fs, subtype='PCM_16')

    # ----- STT: Transcribe the recorded audio -----
    print("Transcribing your response...")
    # Prepare multipart form-data for the STT endpoint
    stt_headers = {
        "api-subscription-key": SARVAM_API_KEY  # Include the subscription key if required
    }
    if language_code is not None and len(language_code) == 2:
        language_code += '-IN'
    if start_conversation:
        language_code = 'unknown'
    elif language_code is None or language_code not in ['hi-IN', 'bn-IN', 'kn-IN', 'ml-IN', 'mr-IN', 'od-IN', 'pa-IN', 'ta-IN', 'te-IN', 'en-IN', 'gu-IN']:
        if language_code is not None and language_code.strip().startswith('en'):
            language_code = 'en-IN'
        else:    
            language_code = 'unknown'
    print(language_code)        
    data = {
        "model": "saarika:v2",
        "language_code": language_code,  # Adjust language code as needed; sample snippet uses "unknown"
    }
    files = {
        "file": (os.path.basename(temp_filename), open(temp_filename, "rb"), "audio/wav")
    }
    stt_response = requests.post(STT_URL, headers=stt_headers, data=data, files=files)
    files["file"][1].close()
    os.remove(temp_filename)
    if stt_response.status_code != 200:
        print("Error with STT:", stt_response.text)
        return None
    stt_json = stt_response.json()
    return stt_json.get("transcript", "")       

def google_stt(language_code, start_conversation):
    # Recording configuration
    silence_duration_required = 3  # Seconds of silence to stop recording
    print(f"Listening for your response (recording will stop after {silence_duration_required} sec of silence)...")
    fs = 44100           # Sample rate (Hz)
    channels = 1         # Mono recording
    chunk_duration = 0.5 # Seconds per chunk
    silence_threshold = 0.01  # RMS amplitude threshold for silence; adjust as needed
    
    recorded_chunks = []
    silence_time = 0

    # Record until we have enough silence
    while True:
        chunk = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=channels, dtype='float32')
        sd.wait()
        recorded_chunks.append(chunk)
        rms = np.sqrt(np.mean(chunk**2))
        if rms < silence_threshold:
            silence_time += chunk_duration
        else:
            silence_time = 0
        if silence_time >= silence_duration_required:
            print("Silence detected. Stopping recording.")
            break

    audio_data = np.concatenate(recorded_chunks, axis=0)
    
    # Save the recorded audio to a temporary WAV file
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_filename = temp_wav.name
    sf.write(temp_filename, audio_data, fs, subtype='PCM_16')

    # Process language code for Indian languages
    if language_code is not None and len(language_code) == 2:
        language_code += '-IN'
    if start_conversation:
        # For Google STT, default to English (or adjust to your desired default)
        language_code = 'en-IN'
    elif language_code is None or language_code not in ['hi-IN', 'bn-IN', 'kn-IN', 'ml-IN', 'mr-IN', 
                                                        'od-IN', 'pa-IN', 'ta-IN', 'te-IN', 'en-IN', 'gu-IN']:
        if language_code is not None and language_code.strip().startswith('en'):
            language_code = 'en-IN'
        else:
            language_code = 'en-IN'
    print(f"Using language code: {language_code}")

    # Transcribe using Google Cloud Speech-to-Text
    client = speech.SpeechClient()
    with io.open(temp_filename, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=fs,
        language_code=language_code,
    )

    print("Transcribing your response...")
    response = client.recognize(config=config, audio=audio)

    # Concatenate the transcription results
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + " "

    # Clean up temporary file
    os.remove(temp_filename)
    
    return transcript.strip()

# =============================================================================
def ask_question(question):
    """
    Parameters:
      - question (str): The question to ask the user.
      
    Returns:
      - user_input (str): The input provided by the user.
    """
    print("\n" + question)
    try:
        user_input = input("> ")
    except (KeyboardInterrupt, EOFError):
        print("\nConversation terminated by the user.")
        exit(0)
    return user_input.strip()


def ask_question_voice(question, language_code = 'unknown', start_conversation = False):
    """
    Parameters:
      - question (str): The question to ask the user.
      - language_code (str): Language in BCP-47 format.
      - start_conversation (bool): First message?
    Returns:
      - transcript (str): The input provided by the user.
    """
    print("Speaking question... : ", question)
    # ----- TTS -----

    # open_ai_tts(question=question)
    sarvam_ai_tts(question=question)

    # ----- STT -----
    #transcript = sarvam_ai_stt(language_code, start_conversation)
    transcript = google_stt(language_code, start_conversation)
    print("Transcription:", transcript)
    return transcript
    

# =============================================================================
# def extract_answered_fields(user_answer, form_fields, llm):
#     """    
#     Parameters:
#       - user_answer (str): The answer provided by the user.
#       - form_fields (list): List of form field dictionaries.
#       - llm: The language model instance.
      
#     Returns:
#       - answered_list (list): List of field_id strings that were answered.
#     """
    
#     #fields_list = ", ".join([field["field_id"] for field in form_fields])

#     field_instructions = "\n".join([
#         f"{field['field_id']}: {field['additional_info']} (Type: {field['datatype']})"
#         for field in form_fields
#     ])
#     prompt = (
#         f"Please Analyze the following answer and determine which of the following field IDs "
#         f"are answered: {field_instructions}.\n\n"
#         f"User Answer: \"{user_answer}\"\n\n"
#         "Respond with a JSON array of the answered field IDs (for example, [\"field_id_1\", \"field_id_2\"])."
#     )

#     messages = [
#         SystemMessage(content="You are an assistant that extracts field IDs from a user's answer."),
#         HumanMessage(content=prompt)
#     ]
#     try:
#         response = llm.invoke(messages).content.strip()
#         # Attempt to parse the JSON response
#         answered_list = loads(response)
#         if not isinstance(answered_list, list):
#             answered_list = []
#     except Exception as e:
#         # Fallback: if the extraction fails, assume the answer addresses the current field.
#         answered_list = []
#     return answered_list



# =============================================================================
def communication(form_fields, language, llm, greeting_message = 'I am Qurate, your personal telecaller assistant.', communication_choice=1):
    """    
    Parameters:
      - form_fields (list): List of form field dictionaries.
      - llm: The language model instance.
      
    Returns:
      - field_parsed_answers (json): List of field_id and their answers.
    """

    # Dictionary to keep track of collected answers keyed by field_id.
    collected_answers = {}
    field_parsed_answers = {field["field_id"]: None for field in form_fields}
    field_asked_counter = {field["field_id"]: 0 for field in form_fields}

    # The chatbot will iterate until all form fields have an associated answer.
    start_converstation = True
    while True:
        if field_parsed_answers.get('language') is not None and field_parsed_answers.get('language') != '':
            language = field_parsed_answers.get('language') 
        # Identify pending fields that have not yet been answered.
        pending_fields = [field for field in form_fields if field_asked_counter.get(field["field_id"]) < 4 and (field_parsed_answers.get(field["field_id"]) is None or field_parsed_answers.get(field["field_id"]) == "")]
        if not pending_fields:
            break  # All questions answered

        # Use previously collected answers to build context for dynamic prompting.
        last_10_conversations = list(collected_answers.items())[-10:]
        #print(last_10_conversations)
        context = "\n".join([f"System: {quest} -> User Response: {ans}" for quest, ans in last_10_conversations])
        # print(context);
        # Choose the first pending field to ask about.
        next_field = pending_fields[0]
        if(language is None or language.strip().startswith('en')):
            language_prompt = 'English'
        else:
            language_prompt = language
        #print(context)
        question_prompt = (
        f"Please generate a relevant and engaging question in {language_prompt} "
        f"(e.g., en-IN for English (Indian Accent), hi-IN for Hindi) that helps collect the field data: {next_field['field_name']}. "
        f"Background: {next_field['additional_info']}. "
        f"Ensure you use simple language and provide the question in {next_field['datatype']} format—only the question itself, with no extra commentary. "
        f"This is attempt {field_asked_counter.get(next_field['field_id'], 0) + 1} for this field. "
        f"If this isn't the first attempt, try a different approach. "
        f"Here is our recent conversation context: {context if context else 'No previous context'}. "
        f"Remember that we have already collected some answers: {field_parsed_answers}. "
        "Feel free to ask follow-up questions or seek clarification if previous responses for current field were unclear. Tone: Show compassion and warmth in your question."
    )

        #print(question_prompt)
        messages = [
            SystemMessage(content=f"You are a conversational human that frames questions naturally."),
            HumanMessage(content=question_prompt)
        ]
        try:
            natural_question = llm.invoke(messages).content.strip()
        except Exception as e:
            # Fallback: if the LLM fails, use a default question.
            natural_question = f"What is your {next_field['field_name']}?"
        
        # Ask the generated question to the user.
        #print(natural_question)
        try:
            if start_converstation:
                natural_question = f"Hello, {greeting_message}. " + natural_question
                #print('If:', natural_question, language, start_converstation)
                if(communication_choice==1):
                    user_response = ask_question(question=natural_question)
                elif(communication_choice==2):  
                    user_response = ask_question_voice(question=natural_question,language_code=language,start_conversation=start_converstation)
                else:
                    user_response = ask_question(question=natural_question)
                #user_response = ask_question_voice(question=natural_question,language_code=language,start_conversation=start_converstation)
                start_converstation = False
            else:
                #print('Else:', natural_question, language, start_converstation)
                if(communication_choice==1):
                    user_response = ask_question(question=natural_question)
                elif(communication_choice==2):  
                    user_response = ask_question_voice(question=natural_question,language_code=language,start_conversation=start_converstation)
                else:
                    user_response = ask_question(question=natural_question)
        except Exception as e:
            print('Inside Exception: ', e)
            break;        
        
        # Use the LLM to extract which fields are addressed in the user's response.
        collected_answers[natural_question] = user_response 
        field_asked_counter[next_field['field_id']] += 1
        field_parsed_answers = parse_for_answers(collected_answers = collected_answers, form_fields = form_fields, llm = llm)
        
         
        # #answered_ids = extract_answered_fields(user_response, [next_field], llm)
        
        # # If the LLM extraction returns nothing, assume the answer is for the current field.
        # if not answered_ids:
        #     answered_ids = [next_field["field_id"]]
        
        # # Save the answer for each detected field.
        # for fid in answered_ids:
        #     collected_answers[fid] = user_response
        
        # (Optional) Show current progress
        print(f"\n[Debug] Current collected answers: {dumps(field_parsed_answers, indent=2)}")
    return field_parsed_answers




# =============================================================================

def parse_for_answers(collected_answers, form_fields, llm):
    """    
    Parameters:
      - collected_answers (map): The answer provided by the user.
      - form_fields (list): List of form field dictionaries.
      - llm: The language model instance.
      
    Returns:
      - parsed_answer (json): List of field_id and their answers
    """
    
    field_instructions = "\n".join([
        f"{field['field_id']}: {field['additional_info']} (Type: {field['datatype']})"
        for field in form_fields
    ])
    today_date = datetime.date.today().strftime("%Y-%m-%d")
    final_prompt = (
        f"Date: {today_date}\n"
        f"Given the concatenated user responses: \"{collected_answers}\",\n"
        f"and the field instructions:\n{field_instructions}\n\n"
        "Extract the information and create a JSON object where each key is a field_id and each value is the user's answer (translated to English). "
        "Correct any typos and include only the relevant part of each answer; if a field is unanswered, set its value to null. "
        "For the 'phone' field, ensure the value is a valid phone number (10-12 digits, numeric only, no letters or special characters). "
        "If the phone number is invalid, set its value to an empty string ('') to indicate it needs re-asking. "
        "For the language field, ensure the value follows the BCP-47 format (e.g., hi-IN, en-IN, bn-IN) since most users are from India. "
        "Return only the JSON object without any additional text or explanation."
    )
    messages = [
        SystemMessage(content="You are an assistant that extracts structured information from text."),
        HumanMessage(content=final_prompt)
    ]
    try:
        json_llm = llm.bind(response_format={"type": "json_object"})
        final_output = json_llm.invoke(messages).content.strip()
        return extract_json(final_output)
    except Exception as e:
        return {}

def parse_for_form_fields(user_query, llm):
    human_message = f"""
        You are provided with a user's text describing the information they want to collect through a form. Your task is to generate a JSON object with a key "fields" that maps to an array of dictionaries. Each dictionary represents a form field with the following keys:
        - "field_id": A unique identifier for the field.
        - "field_name": The display name of the field.
        - "datatype": The type of data expected (e.g., "string").
        - "additional_info": Extra details about what the field is used for.

        Return a JSON object following this exact structure:
        {{
            "fields": [
                {{
                    "field_id": "language",
                    "field_name": "Language",
                    "datatype": "string",
                    "additional_info": "This question is asked so that further communication can happen in that language"
                }},
                {{
                    "field_id": "name",
                    "field_name": "Name",
                    "datatype": "string",
                    "additional_info": "User's full name"
                }}
            ]
        }}

        Important:
        1. If the user's text does not specify a language field, include the language field exactly as shown above as the first element.
        2. If the user's text does specify a language field, ensure that its "field_id" is "language" (regardless of what was provided) and that it appears as the first element.
        3. The remaining fields should be generated based on the details provided in the user's text (for example, name, address, phone number, city, or product interest).

        User Query: {user_query}

        Return the result as valid JSON.
    """

    messages = [
        SystemMessage(content="You are an assistant that extracts structured information from text."),
        HumanMessage(content=human_message)
    ]
    try:
        json_llm = llm.bind(response_format={"type": "json_object"})
        final_output = json_llm.invoke(messages).content.strip()
        return extract_json(final_output)
    except Exception as e:
        return {}

# =============================================================================
def conversation_agent(user_query, communication_choice = 1, language = 'en-IN'):
    """    
    Parameters:
      - form_fields (list): List of form field dictionaries.
      - llm: The language model instance.
    """
           
    llm = select_llm(choice='open_ai',model_name="gpt-4o-mini")
    form_fields = parse_for_form_fields(user_query = user_query, llm = llm).get("fields", [])
    print("\nForm Fields extracted from the user query:")
    print(dumps(form_fields, indent  = 2))
    if not form_fields:
        return
    if form_fields[0].get("field_id") != "language":
            form_fields.insert(0, {
                "field_id": "language",
                "field_name": "Language",
                "datatype": "string",
                "additional_info": "This question is asked so that further communication can happen in that language",
            })        
    final_json = communication(form_fields = form_fields, language = language, llm = llm, greeting_message = 'I your personal telecaller assistant.',communication_choice=communication_choice) 
    print("\nFinal collected responses in JSON format:")
    print(dumps(final_json, indent  = 2))



# =============================================================================
if __name__ == "__main__":
    user_query = "I want to collect language, name, address, phone number, college and age."
    #user_query = str(input("What do you want to collect: "))
    try:  
        communication_choice = int(input("How would you like to communicate: \n1. Chat \n2. Chat (With Audio): This is not aync\n"))
        if communication_choice > 2: 
            print('Invalid Choice so going with chat')
    except:
        print('Invalid Choice so going with chat') 
    conversation_agent(user_query = user_query, communication_choice = communication_choice)
