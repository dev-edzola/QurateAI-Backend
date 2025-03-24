import os
import json
import base64
import audioop
import uuid
import time
import wave
import random
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, send_file, Response, jsonify
from twilio.twiml.voice_response import VoiceResponse, Start, Stop
from twilio.rest import Client
from google.cloud import texttospeech
from google.cloud import speech
from dotenv import load_dotenv
from flask_sock import Sock
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
import numpy as np
import glob
load_dotenv()


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)  # Keep this as is for now
sock = Sock(app)

# Enable CORS for the Flask app
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:8080",           # Local development
            "https://qurate-ai-frontend.onrender.com"  # Production frontend
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure the root logger for file logging (all levels)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(LOG_DIR, 'qurate.log'), 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)

# Create a separate console handler with a higher threshold (INFO only)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# Get the root logger and add the console handler
logger = logging.getLogger('')
logger.addHandler(console_handler)

# Create a custom logger for important application events
app_logger = logging.getLogger('qurate')
app_logger.setLevel(logging.INFO)

# Tracking variables for stream and session management
ACTIVE_STREAMS = {}  # To track active WebSocket connections by call_id
PROCESSED_SESSIONS = {}  # Format: {call_id: {session_id: timestamp}}

# Set up a directory for audio files (absolute path)
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_files")
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Setup directory for transcription audio
TRANSCRIPTION_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "transcription_audio")
if not os.path.exists(TRANSCRIPTION_AUDIO_DIR):
    os.makedirs(TRANSCRIPTION_AUDIO_DIR)

# Initialize Flask app and Flask-Sock for raw WebSocket support
app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
sock = Sock(app)

# Twilio credentials and host settings
account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
twilio_client = Client(account_sid, auth_token)
twilio_phone_number = os.environ.get('TWILIO_PHONE_NUMBER')
host = os.environ.get('HOST')  # e.g., your ngrok or public domain

# OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

# In-memory storage for call state (keyed by call_id)
connected_clients = {}

# Configuration for silence detection (in seconds)
INITIAL_WAIT_TIMEOUT = 10  # Wait 10 seconds for user to start talking
SILENCE_TIMEOUT = 3        # Wait 5 seconds of silence after user stops talking
MIN_RMS_THRESHOLD = 10     # RMS threshold to detect speech
MIN_AUDIO_BUFFER_LENGTH = 1000  # Minimum audio buffer size to process
MAX_AUDIO_DURATION = 30  # Maximum seconds of audio to record before forcing processing
MAX_AUDIO_BYTES = MAX_AUDIO_DURATION * 8000 * 1  # 8kHz, 1 byte per sample for µ-law

# Initialize Text-to-Speech client
tts_client = texttospeech.TextToSpeechClient()

def select_llm(model_name="gpt-4o-mini", provider="open_ai"):
    """Select the language model to use based on the provider."""
    if provider.lower() == "claude":
        return ChatAnthropic(anthropic_api_key=CLAUDE_API_KEY, temperature=0.7, model=model_name)
    else:
        return ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7, model_name=model_name)

llm = select_llm()

# AudioRecorder class for managing audio buffer independently
class AudioRecorder:
    def __init__(self):
        self.buffer = bytearray()
        self.start_time = time.time()
        self.last_voice_time = time.time()
        self.is_speaking = False
        self.rms_values = []
    
    def add_chunk(self, chunk):
        """Add an audio chunk to the buffer and return metrics"""
        self.buffer.extend(chunk)
        
        # Calculate RMS of this chunk to detect speech
        try:
            pcm_chunk = audioop.ulaw2lin(chunk, 2)
            rms = audioop.rms(pcm_chunk, 2)
            self.rms_values.append(rms)
            
            # Update speaking state
            if rms > MIN_RMS_THRESHOLD:
                self.is_speaking = True
                self.last_voice_time = time.time()
                
            return len(self.buffer), rms
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return len(self.buffer), 0
    
    def get_buffer_copy(self):
        """Return a clean copy of the buffer"""
        return bytes(self.buffer)
    
    def clear(self):
        """Clear the buffer and reset state"""
        self.buffer = bytearray()
        self.start_time = time.time()
        self.last_voice_time = time.time()
        self.is_speaking = False
        self.rms_values = []
    
    def get_silence_duration(self):
        """Get seconds of silence since last voice"""
        return time.time() - self.last_voice_time
    
    def get_average_rms(self):
        """Get average RMS of all chunks"""
        if not self.rms_values:
            return 0
        return sum(self.rms_values) / len(self.rms_values)
    
    def get_max_rms(self):
        """Get maximum RMS value"""
        if not self.rms_values:
            return 0
        return max(self.rms_values)
    
    def get_duration(self):
        """Get duration of recording in seconds"""
        return len(self.buffer) / 8000  # 8kHz sampling rate for µ-law

# --- Helper Functions ---
# Function to periodically clean up stale WebSocket connections
def cleanup_stale_streams():
    """Periodically cleanup stale stream connections to prevent resource exhaustion"""
    current_time = time.time()
    streams_to_remove = []
    
    for call_sid, stream_info in ACTIVE_STREAMS.items():
        # Check if this stream has been active for too long (>5 minutes)
        if current_time - stream_info.get("start_time", 0) > 300:
            logger.debug(f"Removing stale stream for call SID {call_sid}, session {stream_info.get('session_id', 'unknown')}")
            # Try to close the WebSocket if it exists
            if "websocket" in stream_info and stream_info["websocket"]:
                try:
                    stream_info["websocket"].close()
                except Exception as e:
                    logger.error(f"Error closing stale WebSocket: {e}")
            streams_to_remove.append(call_sid)
    
    # Remove the stale streams
    for call_sid in streams_to_remove:
        ACTIVE_STREAMS.pop(call_sid, None)
    
    return len(streams_to_remove)

# Health check endpoint for monitoring
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint that also cleans up stale connections"""
    # Clean up any stale connections
    num_removed = cleanup_stale_streams()
    
    # Return basic health information
    return jsonify({
        "status": "ok",
        "active_streams": len(ACTIVE_STREAMS),
        "active_clients": len(connected_clients),
        "stale_streams_removed": num_removed,
        "timestamp": time.time()
    }), 200

# Function to check WebSocket health during audio processing
def check_websocket_health(call_id, session_id):
    """Verify the WebSocket connection is healthy for the current session"""
    client_data = connected_clients.get(call_id)
    if not client_data:
        return False
    
    # Check if this is still the active session
    if str(client_data.get("audio_session_id", 0)) != session_id:
        return False
    
    # Check if the WebSocket is still connected
    call_sid = client_data.get("callSid")
    if not call_sid or call_sid not in ACTIVE_STREAMS:
        return False
    
    stream_info = ACTIVE_STREAMS[call_sid]
    if stream_info.get("session_id") != session_id:
        return False
    
    # Check if the WebSocket object exists and appears valid
    if "websocket" not in stream_info or not stream_info["websocket"]:
        return False
    
    return True

def save_audio_for_transcription(call_id, audio_data, transcription, session_id=0, timestamp=None):
    """
    Save the audio data to a file with the transcription as metadata.
    
    Args:
        call_id (str): The unique identifier for the call
        audio_data (bytes): The µ-law encoded audio data
        transcription (str): The transcription text
        session_id (int): The audio session ID to track separate recordings
        timestamp (int): Optional timestamp to ensure unique filenames
    
    Returns:
        dict: Information about the saved files
    """
    # Generate a timestamp - use provided timestamp or current time
    if timestamp:
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
        ms = int((timestamp % 1) * 1000)  # Add milliseconds for extra uniqueness
        timestamp_str = f"{ts}-{ms:03d}"
    else:
        timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    
    # Create a filename with the call_id, timestamp, and a short version of the transcription
    # Limit the transcription to 30 chars for the filename and remove problematic characters
    short_transcription = transcription[:30].replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_transcription = ''.join(c for c in short_transcription if c.isalnum() or c == '_')
    
    # Generate a unique filename that includes session ID and timestamp
    wav_filename = f"{call_id}_s{session_id}_{timestamp_str}_{safe_transcription}.wav"
    json_filename = f"{call_id}_s{session_id}_{timestamp_str}_{safe_transcription}.json"
    ulaw_filename = f"{call_id}_s{session_id}_{timestamp_str}_{safe_transcription}.ulaw"
    
    # Full paths
    wav_path = os.path.join(TRANSCRIPTION_AUDIO_DIR, wav_filename)
    json_path = os.path.join(TRANSCRIPTION_AUDIO_DIR, json_filename)
    ulaw_path = os.path.join(TRANSCRIPTION_AUDIO_DIR, ulaw_filename)
    
    try:
        # First save the original µ-law data for reference (helps with debugging)
        if isinstance(audio_data, bytearray):
            audio_data = bytes(audio_data)
        
        with open(ulaw_path, 'wb') as ulaw_file:
            ulaw_file.write(audio_data)
        
        # Convert µ-law to linear PCM with careful error handling
        try:
            pcm_data = audioop.ulaw2lin(audio_data, 2)
        except Exception as e:
            logger.error(f"Error converting audio to PCM: {e}")
            # If conversion fails, try again with a clean copy
            try:
                pcm_data = audioop.ulaw2lin(bytes(audio_data), 2)
            except Exception as e2:
                logger.error(f"Second attempt at PCM conversion failed: {e2}")
                # Create an empty PCM buffer as a last resort
                pcm_data = b'\x00\x00' * (len(audio_data))
        
        # Save as WAV file (8000 Hz, 16-bit mono)
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(8000)
            wav_file.writeframes(pcm_data)
        
        # Calculate audio quality metrics for debugging
        if len(pcm_data) > 0:
            try:
                rms = audioop.rms(pcm_data, 2)
                max_sample = audioop.max(pcm_data, 2)
                min_sample = audioop.minmax(pcm_data, 2)[0]  # Get the minimum value
                
                # Calculate average value safely
                if isinstance(pcm_data, bytes):
                    # For bytes objects, we need to manually compute the average
                    sum_value = 0
                    for i in range(0, len(pcm_data), 2):
                        if i+1 < len(pcm_data):
                            # Convert each 16-bit sample (little-endian) to signed int
                            sample = pcm_data[i] + (pcm_data[i+1] << 8)
                            # Handle signed values (if high bit is set, it's negative)
                            if sample > 32767:
                                sample -= 65536
                            sum_value += sample
                    avg_value = sum_value / (len(pcm_data) // 2) if len(pcm_data) > 0 else 0
                else:
                    avg_value = sum(pcm_data) / len(pcm_data) if len(pcm_data) > 0 else 0
            except Exception as e:
                logger.error(f"Error calculating audio metrics: {e}")
                rms, max_sample, min_sample, avg_value = 0, 0, 0, 0
        else:
            rms, max_sample, min_sample, avg_value = 0, 0, 0, 0
        
        # Save metadata as JSON with audio quality metrics
        metadata = {
            "call_id": call_id,
            "session_id": session_id,
            "timestamp": timestamp_str,
            "transcription": transcription,
            "audio_duration_seconds": len(pcm_data) / (8000 * 2),
            "language": connected_clients.get(call_id, {}).get("language", "unknown"),
            "audio_metrics": {
                "rms": rms,
                "max_sample": max_sample,
                "min_sample": min_sample,
                "average_value": float(avg_value),
                "original_ulaw_size": len(audio_data),
                "pcm_size": len(pcm_data)
            }
        }
        
        with open(json_path, 'w') as json_file:
            json.dump(metadata, json_file, indent=2)
        
        logger.debug(f"Saved audio files for session {session_id} at {timestamp_str}")
        
        return {
            "wav_path": wav_path,
            "ulaw_path": ulaw_path,
            "json_path": json_path,
            "transcription": transcription,
            "audio_metrics": metadata["audio_metrics"]
        }
    except Exception as e:
        logger.error(f"Error saving audio file: {e}")
        return None



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



def cleanup_call_audio_files(call_id):
    """Delete all audio files associated with a specific call_id"""
    logger.info(f"Cleaning up audio files for call {call_id}")
    
    files_deleted = 0
    
    # 1. Clean up TTS response files
    response_pattern = os.path.join(AUDIO_DIR, f"response_{call_id}_*.mp3")
    for file_path in glob.glob(response_pattern):
        try:
            os.remove(file_path)
            files_deleted += 1
        except Exception as e:
            logger.error(f"Error deleting audio file {file_path}: {e}")
    
    # 2. Clean up transcription audio files
    # Look for all file types (.wav, .ulaw, .json) with this call_id
    for file_type in ['.wav', '.ulaw', '.json']:
        transcription_pattern = os.path.join(TRANSCRIPTION_AUDIO_DIR, f"{call_id}*{file_type}")
        for file_path in glob.glob(transcription_pattern):
            try:
                os.remove(file_path)
                files_deleted += 1
            except Exception as e:
                logger.error(f"Error deleting transcription file {file_path}: {e}")
    
    logger.info(f"Deleted {files_deleted} files for call {call_id}")
    return files_deleted

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

def parse_for_answers(collected_answers, form_fields, llm):
    """Parse collected answers to extract structured data"""
    field_instructions = "\n".join([
        f"{field['field_id']}: {field['additional_info']} (Type: {field['datatype']})"
        for field in form_fields
    ])
    final_prompt = (
        f"Given the collected conversation: {collected_answers},\n"
        f"and the field instructions:\n{field_instructions}\n\n"
        "Extract the information and create a JSON object where each key is a field_id and each value is the user's answer (translated to English). "
        "Correct any typos and include only the relevant part of each answer; if a field is unanswered, set its value to null. "
        "For the language field, ensure the value follows the BCP-47 format (e.g., hi-IN, en-IN, bn-IN) since most users are from India. "
        "Return only the JSON object without any additional text or explanation."
    )
    messages = [
        SystemMessage(content="You are an assistant that extracts structured information from text."),
        HumanMessage(content=final_prompt)
    ]
    try:
        json_llm = llm
        final_output = json_llm.invoke(messages).content.strip()
        parsed_data = extract_json(final_output)
        print(f"\n[Debug] Current collected answers: {json.dumps(parsed_data, indent=2)}")
        return parsed_data
    except Exception as e:
        logger.error(f"Error parsing answers: {e}")
        return {}

def get_next_question(form_fields, collected_answers, field_parsed_answers, field_asked_counter, llm, language="en-GB", greeting_message=None, call_id=None):
    """Generate the next question based on collected answers and question attempts"""
    pending_fields = [
        field for field in form_fields 
        if field_asked_counter.get(field["field_id"], 0) < 4 and 
           (field_parsed_answers.get(field["field_id"]) is None or field_parsed_answers.get(field["field_id"]) == "")
    ]


    language_prompt = language if language != None and not language.lower().startswith('en') else "English"


    if not pending_fields:
        # Modified to just return a simple thank-you without asking for more
        cleanup_call_audio_files(call_id)
        return None, generate_summary_response(field_parsed_answers, form_fields, llm, language=language_prompt)
    
    next_field = pending_fields[0]
    # Use provided language if available; otherwise, default to "English"
    
    last_10_conversations = list(collected_answers.items())[-10:]
    context = "\n".join([f"System: {quest} -> User Response: {ans}" for quest, ans in last_10_conversations])
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
 
    messages = [
        SystemMessage(content="You are a conversational human that frames questions naturally."),
        HumanMessage(content=question_prompt)
    ]
    try:
        natural_question = llm.invoke(messages).content.strip()
        if ":" in natural_question:
            natural_question = natural_question.split(":", 1)[1].strip()
        else:
            natural_question = natural_question
        if greeting_message and not collected_answers:
            natural_question = f"Hello, {greeting_message}. {natural_question}"
        
        return next_field["field_id"], natural_question
    except Exception as e:
        logger.error(f"LLM error in get_next_question: {e}")
        return next_field["field_id"], f"What is your {next_field['field_name']}?"

def detect_silence(audio_data, sample_rate=8000, chunk_duration=0.5,
                  threshold=8, silence_duration_required=3.0):
    """Enhanced silence detection optimized for Indian languages."""
    try:
        # Convert µ-law encoded audio to linear PCM (2 bytes per sample)
        pcm_data = audioop.ulaw2lin(audio_data, 2)
    except Exception as e:
        logger.error(f"Error converting audio to PCM: {e}")
        return False, None, []

    sample_width = 2  # bytes per sample
    chunk_size = int(chunk_duration * sample_rate * sample_width)
    
    silence_time = 0.0
    offset = 0
    rms_values = []
    silence_segments = []
    silence_start = None

    while offset < len(pcm_data):
        chunk = pcm_data[offset:offset+chunk_size]
        if not chunk:
            break
        
        try:
            rms = audioop.rms(chunk, sample_width)
        except Exception as e:
            logger.error(f"Error computing RMS: {e}")
            rms = 0
        rms_values.append(rms)

        # Current time position in seconds
        current_time = offset / (sample_rate * sample_width)
        
        if rms < threshold:
            # Start tracking a new silence segment
            if silence_start is None:
                silence_start = current_time
            
            silence_time += chunk_duration
        else:
            # End of a silence segment
            if silence_start is not None:
                silence_segments.append((silence_start, current_time))
                silence_start = None
            
            silence_time = 0.0

        # Check if we've reached our required silence duration
        if silence_time >= silence_duration_required:
            # Make sure to record the final silence segment
            if silence_start is not None:
                silence_segments.append((silence_start, current_time))
            
            overall_avg_rms = np.mean(rms_values) if rms_values else 0
            return True, overall_avg_rms, silence_segments

        offset += chunk_size

    # Make sure to record the final silence segment if we ended during silence
    if silence_start is not None:
        current_time = len(pcm_data) / (sample_rate * sample_width)
        silence_segments.append((silence_start, current_time))
    
    overall_avg_rms = np.mean(rms_values) if rms_values else 0
    return False, overall_avg_rms, silence_segments

def proper_language_code(language_code, type = "speech"):

    if language_code is None or language_code.lower().startswith("en"):
        if type == "speech":
            return "en-IN"
        elif type == "transcription":
            return "en-GB"
        else:
            return "en-GB"

    language_mapping = {
        "en": "en-IN",  # Default to English
        "hi": "hi-IN",  # Hindi
        "mr": "mr-IN",  # Marathi
        "bn": "bn-IN",  # Bengali
        "ta": "ta-IN",  # Tamil
        "te": "te-IN",  # Telugu
        "kn": "kn-IN",  # Kannada
        "ml": "ml-IN",  # Malayalam
        "gu": "gu-IN",  # Gujarati
        "pa": "pa-IN",  # Punjabi
    }
    

    if language_code.lower() == "english":
        language_code = "en-GB"
    elif language_code.lower() == "hindi":
        language_code = "hi-IN"
    elif language_code.lower() == "marathi":
        language_code = "mr-IN"
    elif language_code.lower() == "tamil":
        language_code = "ta-IN"
    elif language_code.lower() == "bengali":
        language_code = "bn-IN"
    elif language_code.lower() == "telugu":
        language_code = "te-IN"
    elif language_code.lower() == "malayalam":
        language_code = "ml-IN"
    elif language_code.lower() == "gujarati":
        language_code = "gu-IN"
    elif language_code.lower() == "kannada":
        language_code = "kn-IN"
    elif language_code.lower() == "punjabi":
        language_code = "pa-IN"
    elif language_code.lower() == "urdu":
        language_code = "ur-IN"

    # Check if we need to map a simple language code to a full one
    if language_code in language_mapping:
        language_code = language_mapping[language_code]
        
    return language_code

def transcribe_audio(audio_data, language_code="en-GB", skip_mapping=False):
    """Enhanced function to transcribe audio with improved handling for Indian languages"""
    try:
        # Normalize language code format for Google Speech API
        
        language_hints = [
            "en-IN",  # English (Indian accent)
            "en-GB",  # English (UK accent)
            "en-US",  # English (US accent)
            "hi-IN",  # Hindi
            "ta-IN",  # Tamil
            "bn-IN",  # Bengali
            "te-IN",  # Telugu
            "ml-IN",  # Malayalam
            "gu-IN",  # Gujarati
            "kn-IN",  # Kannada
            "mr-IN",  # Marathi
            "pa-IN",  # Punjabi
            "ur-IN"  # Urdu
        ]
        
        # Handle common language name inputs
        if not skip_mapping:
            language_code = proper_language_code(language_code, type="transcription")
            
        logger.debug(f"Attempting speech recognition with language code: {language_code}")
        
        speech_client = speech.SpeechClient()
        speech_audio = speech.RecognitionAudio(content=audio_data)
        language_hints.remove(language_code)

        print("Transcription Language:", language_code)
        # Define base configuration
        config_params = {
            "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "sample_rate_hertz": 8000,
            "language_code": language_code,
            "enable_automatic_punctuation": True,
            "use_enhanced": True,
            "model": "latest_long",  # Use the latest long model for better transcription
            "audio_channel_count": 1,  # Mono audio from phone calls
            "alternative_language_codes": language_hints,  # Additional language hints
        }
        
        # For Indian languages, add speech adaptation for better recognition
        if language_code.endswith("-IN"):
            speech_contexts = []
            
            # Common words in various Indian languages to help with recognition
            if language_code == "hi-IN":
                speech_contexts.append(speech.SpeechContext(
                    phrases=["हां", "नहीं", "धन्यवाद", "नमस्ते", "आप", "मैं", "हम"]
                ))
            elif language_code == "mr-IN":
                speech_contexts.append(speech.SpeechContext(
                    phrases=["हो", "नाही", "धन्यवाद", "नमस्कार", "तुम्ही", "मी", "आम्ही"]
                ))
            
            # For Indian English specifically, include both US and Indian English models
            if language_code != None and language_code.lower().startswith("en"):
                # config_params["alternative_language_codes"] = ["en-IN"]
                speech_contexts.append(speech.SpeechContext(
                    phrases=["yes", "no", "thank you", "hello", "I am", "my name is"]
                ))
            
            if speech_contexts:
                config_params["speech_contexts"] = speech_contexts
        
        speech_config = speech.RecognitionConfig(**config_params)
        
        # For longer audio segments, use long running recognition
        audio_length = len(audio_data) / (8000 * 2)  # Approximate length in seconds
        
        if audio_length > 30:  # For audio longer than 30 seconds
            operation = speech_client.long_running_recognize(config=speech_config, audio=speech_audio)
            logger.debug("Waiting for long-running operation to complete...")
            response = operation.result(timeout=90)
        else:
            # For shorter clips, use standard recognition
            response = speech_client.recognize(config=speech_config, audio=speech_audio)
            
        # Get transcript with confidence scores
        transcription = ""
        avg_confidence = 0
        result_count = 0
        
        for result in response.results:
            if result.alternatives:
                transcription += result.alternatives[0].transcript + " "
                avg_confidence += result.alternatives[0].confidence
                result_count += 1
        
        if result_count > 0:
            avg_confidence /= result_count
            app_logger.info(f"Speech recognized with {language_code}: '{transcription.strip()}' (Avg confidence: {avg_confidence:.2f})")
        
        return transcription.strip()
    except Exception as e:
        logger.error(f"Speech recognition error with language {language_code}: {e}")
        return ""

# --- Processing Functions ---

def process_user_audio(call_id, call_sid, session_id):
    """Process the user's audio buffer with deduplication protection"""
    if call_id not in connected_clients:
        logger.error(f"Cannot process audio for unknown call_id: {call_id}")
        return
    
    client_data = connected_clients[call_id]
    
    # Validate session
    if str(client_data.get("audio_session_id", 0)) != session_id:
        logger.debug(f"Ignoring audio processing for outdated session {session_id}")
        return
    
    # DEDUPLICATION CHECK: Check if this session was already processed
    if call_id in PROCESSED_SESSIONS and session_id in PROCESSED_SESSIONS[call_id]:
        last_processed = PROCESSED_SESSIONS[call_id][session_id]
        # Only allow reprocessing if more than 10 seconds have passed (safety valve)
        if time.time() - last_processed < 10:
            logger.debug(f"Skipping duplicate processing for call {call_id} session {session_id} (processed {time.time() - last_processed:.1f}s ago)")
            return
    
    # Set processing flag to prevent concurrent processing
    if client_data.get("processing", False):
        logger.debug(f"Already processing audio for call {call_id}")
        return
    
    client_data["processing"] = True
    
    try:
        # Get the audio recorder
        audio_recorder = client_data.get("audio_recorder")
        if not audio_recorder or len(audio_recorder.buffer) == 0:
            logger.debug(f"Empty audio buffer for call {call_id}")
            client_data["processing"] = False
            return
        
        # Record that we're processing this session
        if call_id not in PROCESSED_SESSIONS:
            PROCESSED_SESSIONS[call_id] = {}
        PROCESSED_SESSIONS[call_id][session_id] = time.time()
        
        # Get a clean copy of the audio buffer
        original_audio_buffer = audio_recorder.get_buffer_copy()
        
        # Get audio metrics for logging
        buffer_length = len(original_audio_buffer)
        audio_duration = audio_recorder.get_duration()
        
        logger.debug(f"Processing audio for call {call_id} session {session_id}: " +
                    f"Buffer size: {buffer_length} bytes, Duration: {audio_duration:.2f} seconds")
        
        # Get the current language or use default
        language_code = client_data.get("language")
        
        # Convert µ-law to PCM for speech recognition
        try:
            pcm_data = audioop.ulaw2lin(original_audio_buffer, 2)
        except Exception as e:
            logger.error(f"Error converting audio to PCM: {e}")
            client_data["processing"] = False
            return
        
        # Try to transcribe with the specified language
        transcription = transcribe_audio(pcm_data, language_code)
        
        # If transcription failed and it's not English, try with English as fallback
        if not transcription and language_code != "en-IN":
            logger.debug(f"No transcription with {language_code}. Trying with en-IN as fallback...")
            transcription = transcribe_audio(pcm_data, "en-IN", skip_mapping=True)
        
        # Update the client data with the transcription
        if transcription:
            app_logger.info(f"Transcription for call {call_id} session {session_id}: '{transcription}'")
            
            # Generate a unique filename using timestamp to prevent overwriting
            timestamp = int(time.time())
            
            # Save the audio file with the transcription
            save_result = save_audio_for_transcription(
                call_id=call_id, 
                audio_data=original_audio_buffer, 
                transcription=transcription,
                session_id=int(session_id) if session_id.isdigit() else 0,
                timestamp=timestamp  # Pass timestamp to ensure uniqueness
            )
            
            # Update collected answers
            current_message = client_data.get("current_message", "")
            if current_message and client_data["current_state"] == "asking":
                client_data["collected_answers"][current_message] = transcription
                app_logger.info(f"Q: {current_message} A: {transcription}")
                
                # Update the current field's parsed answer if we're in asking state
               
                try:
                        parsed_data = parse_for_answers(
                            client_data["collected_answers"],
                            client_data["form_fields"],
                            llm=llm
                        )
                        client_data["field_parsed_answers"] = parsed_data
                except Exception as e:
                    logger.error(f"Error parsing answers: {e}")
                    


                # if "current_field" in client_data:
                #     current_field = client_data["current_field"]
                #     client_data["field_parsed_answers"][current_field] = transcription
                
                # Move to next question state
                client_data["current_state"] = "next_question"
            else:
                # If we're not in asking state, just store the transcription
                client_data["last_transcription"] = transcription
        else:
            logger.debug(f"No transcription for call {call_id} session {session_id}")
    
    except Exception as e:
        logger.error(f"Error processing audio for call {call_id}: {e}")
    finally:
        # IMPORTANT: Make sure we release the processing lock
        client_data["processing"] = False
        client_data["last_processed"] = time.time()
        
        # Create a new AudioRecorder to ensure we start fresh for next recording
        client_data["audio_recorder"] = AudioRecorder()
        
        # Redirect to voice webhook to get the next question or response
        try:
            if client_data.get("callSid"):
                twilio_client.calls(client_data["callSid"]).update(
                    url=f"https://{host}/voice?call_id={call_id}",
                    method="POST"
                )
        except Exception as e:
            logger.error(f"Error redirecting call {call_id}: {e}")

def process_with_empty_response(call_id, call_sid, session_id):
    """Process an empty response when user doesn't speak, with deduplication protection"""
    if call_id not in connected_clients:
        return
        
    client_data = connected_clients[call_id]
    
    # Validate session
    if str(client_data.get("audio_session_id", 0)) != session_id:
        logger.debug(f"Ignoring empty response processing for outdated session {session_id}")
        return
    
    # Deduplication check
    if call_id in PROCESSED_SESSIONS and session_id in PROCESSED_SESSIONS[call_id]:
        last_processed = PROCESSED_SESSIONS[call_id][session_id]
        if time.time() - last_processed < 10:
            logger.debug(f"Skipping duplicate empty response for call {call_id} session {session_id}")
            return
    
    # Set processing flag to prevent concurrent processing
    if client_data.get("processing", False):
        return
        
    client_data["processing"] = True
    
    try:
        # Record that we're processing this session
        if call_id not in PROCESSED_SESSIONS:
            PROCESSED_SESSIONS[call_id] = {}
        PROCESSED_SESSIONS[call_id][session_id] = time.time()
        
        # Empty transcription indicates no response
        client_data["last_transcription"] = ""
        
        # Move to next question state
        client_data["current_state"] = "next_question"
        
        logger.debug(f"Processed empty response for call {call_id} session {session_id}")
    except Exception as e:
        logger.error(f"Error processing empty response for call {call_id}: {e}")
    finally:
        client_data["processing"] = False
        
        # Create a new AudioRecorder to ensure we start fresh
        client_data["audio_recorder"] = AudioRecorder()
        
        # Redirect to voice webhook to get the next question
        try:
            if client_data.get("callSid"):
                twilio_client.calls(client_data["callSid"]).update(
                    url=f"https://{host}/voice?call_id={call_id}",
                    method="POST"
                )
        except Exception as e:
            logger.error(f"Error redirecting call {call_id}: {e}")

def handle_ws_message(call_id, message, session_id):
    """Process incoming WebSocket messages with deduplication protection and reduced logging"""
    client_data = connected_clients[call_id]
    
    # Double-check session - ignore messages for outdated sessions (without spamming logs)
    if str(client_data.get("audio_session_id", 0)) != session_id:
        # We don't log here since the parent function already logs outdated sessions
        return
    
    # Skip if this session is already processed (check PROCESSED_SESSIONS)
    if (call_id in PROCESSED_SESSIONS and 
        session_id in PROCESSED_SESSIONS[call_id] and 
        time.time() - PROCESSED_SESSIONS[call_id][session_id] < 10):
        # Only log occasionally to avoid flooding logs
        if random.random() < 0.001:  # Log only ~0.1% of skipped messages
            logger.debug(f"Ignoring messages for already processed session {session_id}")
        return

    # Only process incoming audio if the system is awaiting a user response
    if not client_data.get("awaiting_response", False):
        return

    try:
        data = json.loads(message)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding WebSocket message for call {call_id}: {e}")
        return

    event = data.get("event")
    if event == "media":
        payload = data["media"]["payload"]
        try:
            # Decode the base64 audio data
            audio_data = base64.b64decode(payload)
            
            # Get the current audio session information
            audio_recorder = client_data.get("audio_recorder")
            
            # If the recorder doesn't exist or is None, create a new one
            if not audio_recorder:
                logger.debug(f"Creating new AudioRecorder for call {call_id} (session {session_id})")
                audio_recorder = AudioRecorder()
                client_data["audio_recorder"] = audio_recorder
            
            # Add the audio chunk to the recorder and get updated buffer size
            buffer_size, rms = audio_recorder.add_chunk(audio_data)
            
            # Periodically log audio progress for debugging (reduce frequency)
            if buffer_size % 32000 == 0:  # Log every ~4 seconds instead of 2
                silence_duration = audio_recorder.get_silence_duration()
                logger.debug(f"Audio progress - Call {call_id} Session {session_id}: {buffer_size/8000:.1f}s, "
                          f"Silence: {silence_duration:.1f}s, RMS: {rms}")
            
            # Check if we've exceeded max buffer size - process it regardless of silence
            if buffer_size > MAX_AUDIO_BYTES and not client_data.get("processing", False):
                logger.debug(f"Audio buffer exceeded maximum size ({MAX_AUDIO_BYTES} bytes). Processing audio.")
                process_user_audio(call_id, data.get("callSid", ""), session_id)
                return
            
            # Current time information
            current_time = time.time()
            time_since_question = current_time - client_data.get("question_time", 0)
            
            # CASE 1: User hasn't started speaking yet
            if not audio_recorder.is_speaking:
                # If user hasn't started speaking within INITIAL_WAIT_TIMEOUT seconds, move on
                if time_since_question > INITIAL_WAIT_TIMEOUT and not client_data.get("processing", False):
                    logger.debug(f"User did not speak for {INITIAL_WAIT_TIMEOUT} seconds after question. Moving to next question.")
                    process_with_empty_response(call_id, data.get("callSid", ""), session_id)
            
            # CASE 2: User has spoken but has been silent for a while
            else:
                silence_duration = audio_recorder.get_silence_duration()
                
                # If silence threshold reached, process the audio
                # Make sure we're not already processing and this session hasn't been processed yet
                if (silence_duration > SILENCE_TIMEOUT and 
                    not client_data.get("processing", False) and
                    not (call_id in PROCESSED_SESSIONS and session_id in PROCESSED_SESSIONS[call_id])):
                    
                    logger.debug(f"Detected {SILENCE_TIMEOUT}s of silence after user spoke. Processing audio.")
                    process_user_audio(call_id, data.get("callSid", ""), session_id)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for call {call_id}: {e}")

# --- HTTP Endpoints ---
@app.route('/make-call', methods=['POST'])
def make_call():
    """Initiate a Twilio phone call with dynamically generated form fields based on user query"""
    to_number = request.form.get('to', '')
    if not to_number:
        return "Please provide a 'to' phone number", 400
    
    # Get user query for form fields generation
    user_query = request.form.get('query', '')
    
    # Parse form fields from user query
    if user_query:
        try:
            form_fields_data = parse_for_form_fields(user_query, llm)
            form_fields = form_fields_data.get('fields', [])
            if not form_fields:
                raise ValueError("No form fields generated")
        except Exception as e:
            logger.error(f"Error parsing form fields from query: {e}")
            form_fields = get_default_form_fields()
    else:
        # Use default form fields if no query provided
        form_fields = get_default_form_fields()
    
    call_id = str(uuid.uuid4())
    connected_clients[call_id] = {
        "audio_recorder": AudioRecorder(),  # New dedicated audio recorder
        "current_state": "initial",
        "callSid": None,
        "last_voice": time.time(),
        "processing": False,
        "last_processed": 0,
        "form_fields": form_fields,
        "collected_answers": {},
        "field_parsed_answers": {field["field_id"]: None for field in form_fields},
        "field_asked_counter": {field["field_id"]: 0 for field in form_fields},
        "language": "en-IN",  # Default language set to Indian English
        "greeting_message": "I am your personal telecaller assistant.",
        "awaiting_response": False,  # Initially false while system is speaking
        "question_time": 0,         # Time when the system finished asking a question
        "max_wait_time": INITIAL_WAIT_TIMEOUT,  # Maximum time to wait for user to start talking
        "silence_timeout": SILENCE_TIMEOUT,     # Time of silence to wait after user stops talking
        "audio_session_id": 0,      # Incremented for each new audio session to ensure isolation
    }
    
    # Log the generated form fields for debugging
    app_logger.info(f"Generated form fields for call {call_id}: {json.dumps(form_fields)}")
    
    call = twilio_client.calls.create(
        to=to_number,
        from_=twilio_phone_number,
        url=f"https://{host}/voice?call_id={call_id}"
    )
    app_logger.info(f"Call initiated to {to_number} with call_id: {call_id}")
    return jsonify({
        "message": f"Call initiated with SID: {call.sid}",
        "call_id": call_id,
        "form_fields": form_fields
    }), 200

def parse_for_form_fields(user_query, llm):
    """
    Parse user query to generate form fields using LLM.
    Returns a dictionary with a 'fields' key containing the field definitions.
    """
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
        # Use response_format for OpenAI models
        try:
            json_llm = llm
            if hasattr(llm, 'bind') and callable(getattr(llm, 'bind')):
                json_llm = llm.bind(response_format={"type": "json_object"})
        except Exception as e:
            logger.warning(f"Could not set response_format for LLM: {e}")
            
        final_output = json_llm.invoke(messages).content.strip()
        return extract_json(final_output)
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
        },
        {
            "field_id": "name",
            "field_name": "Name",
            "datatype": "string",
            "additional_info": "User's full name"
        },
        {
            "field_id": "address",
            "field_name": "Address",
            "datatype": "string",
            "additional_info": "User's residential address"
        },
        {
            "field_id": "phone",
            "field_name": "Phone Number",
            "datatype": "string",
            "additional_info": "User's contact phone number"
        },
        {
            "field_id": "city",
            "field_name": "City",
            "datatype": "string",
            "additional_info": "City of residence"
        },
        {
            "field_id": "product_interest",
            "field_name": "Product Interest",
            "datatype": "string",
            "additional_info": "The product(s) the user is interested in"
        }
    ]


@app.route('/voice', methods=['POST'])
def voice():
    """Handle Twilio voice webhook with aggressive WebSocket management"""
    call_id = request.args.get('call_id')
    call_sid = request.form.get('CallSid')
    logger.debug(f"Received voice webhook for call {call_id} with SID: {call_sid}")
    
    if not call_id or call_id not in connected_clients:
        response = VoiceResponse()
        response.say("Sorry, there was an error with this call. Will try to reach you again.")
        response.hangup()
        return Response(response.to_xml(), mimetype='text/xml')
    
    client_data = connected_clients[call_id]
    if not client_data.get("callSid") and call_sid:
        client_data["callSid"] = call_sid
    
    client_data["last_voice"] = time.time()
    
    # Set flag to indicate system is speaking
    client_data["awaiting_response"] = False
    
    # Create a new audio recorder for each question
    client_data["audio_recorder"] = AudioRecorder()
    
    # Increment the audio session ID
    client_data["audio_session_id"] = client_data.get("audio_session_id", 0) + 1
    current_session = client_data["audio_session_id"]
    
    logger.debug(f"Starting new audio session {current_session} for call {call_id}")
    
    # IMPORTANT: Close any existing WebSockets for this call to free up resources
    if call_sid in ACTIVE_STREAMS:
        try:
            old_stream = ACTIVE_STREAMS[call_sid]
            if "websocket" in old_stream and old_stream["websocket"]:
                logger.debug(f"Forcibly closing WebSocket for previous session {old_stream.get('session_id')}")
                try:
                    old_stream["websocket"].close()
                except Exception as e:
                    logger.error(f"Error while closing WebSocket: {e}")
            ACTIVE_STREAMS.pop(call_sid, None)
        except Exception as e:
            logger.error(f"Error cleaning up old streams: {e}")
    
    # Determine which question to ask based on the current state
    if "language" in client_data and "language" in client_data["field_parsed_answers"]:
        client_data["language"] = proper_language_code(client_data["field_parsed_answers"]["language"], type="question")
    
    if client_data["current_state"] in ["initial", "next_question"]:
        logger.debug(f"Current State: {client_data['current_state']}")
        print("Question Language: ", client_data.get("language"))
        field_id, message = get_next_question(
            client_data["form_fields"], 
            client_data["collected_answers"],
            client_data["field_parsed_answers"],
            client_data["field_asked_counter"],
            llm,
            client_data.get("language"),
            client_data.get("greeting_message") if client_data["current_state"] == "initial" else None,
            call_id=call_id
        )
        
        if field_id is None:
            client_data["current_state"] = "complete"
        else:
            client_data["current_field"] = field_id
            client_data["current_state"] = "asking"
            client_data["field_asked_counter"][field_id] += 1
            client_data["current_message"] = message
    elif client_data["current_state"] == "complete":
        # Modified to just say thank you without asking for more
        message = generate_summary_response(client_data["field_parsed_answers"], client_data["form_fields"], llm, language=client_data.get("language"))
        cleanup_call_audio_files(call_id)
    else:
        message = client_data.get("current_message", "Please continue.")
    
    # Check if we need to hang up the call
    should_hangup = False
    if client_data["current_state"] == "complete":
        # Modified to hang up immediately when complete, without waiting for response
        should_hangup = True
        message = generate_summary_response(client_data["field_parsed_answers"], client_data["form_fields"], llm, language=client_data.get("language"))
        cleanup_call_audio_files(call_id)
    
    # Synthesize the message using Google TTS
    speech_language = proper_language_code(client_data.get("language"), type="speech")
    print('Speech Language: ', speech_language)
    synthesis_input = texttospeech.SynthesisInput(text=message)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=speech_language,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    
    try:
        tts_response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        logger.debug(f"TTS synthesis successful for voice endpoint - session {current_session}")
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        tts_response = None

    if tts_response:
        audio_file_path = os.path.join(AUDIO_DIR, f"response_{call_id}_{current_session}.mp3")
        try:
            with open(audio_file_path, "wb") as out:
                out.write(tts_response.audio_content)
            logger.debug(f"TTS audio file written to {audio_file_path}")
        except Exception as e:
            logger.error(f"Error writing TTS audio file: {e}")

    # Build TwiML response
    response = VoiceResponse()
    
    # If we need to hang up, just play the goodbye message and hang up
    if should_hangup:
        response.play(f"https://{host}/audio/{call_id}/{current_session}")
        response.hangup()
        app_logger.info(f"Hanging up call {call_id} after final message")
    else:
        # IMPORTANT: First add a <Stop> verb to stop any existing streams
        # This is the correct way to stop streams, not response.stop_stream()
        response.append(Stop())
        logger.debug(f"Adding Stop verb to end any existing streams for call {call_sid}")
        
        # Then start a new stream with a unique URL that includes the session ID
        start = Start()
        stream_url = f"wss://{host}/media/{call_id}/{current_session}"
        start.stream(url=stream_url)
        response.append(start)
        
        # Track this new stream (but don't set websocket yet - that happens when the WebSocket connects)
        ACTIVE_STREAMS[call_sid] = {
            "url": stream_url,
            "session_id": current_session,
            "start_time": time.time()
        }
        
        # Play the TTS audio
        response.play(f"https://{host}/audio/{call_id}/{current_session}")
        
        # Redirect after playback to set the flag allowing user input
        response.redirect(f"https://{host}/set-awaiting-response?call_id={call_id}&session_id={current_session}")
    
    app_logger.info(f"Question asked: {message}")
    return Response(response.to_xml(), mimetype='text/xml')

@app.route('/set-awaiting-response', methods=['GET', 'POST'])
def set_awaiting_response():
    """Callback endpoint to set the flag after audio playback completes."""
    call_id = request.args.get('call_id')
    session_id = request.args.get('session_id', '0')
    
    if call_id and call_id in connected_clients:
        # Set flags and timing information
        client_data = connected_clients[call_id]
        
        # Verify this is still the current session
        current_session = str(client_data.get("audio_session_id", 0))
        if session_id != current_session:
            logger.debug(f"Warning: Received outdated awaiting-response for session {session_id}, current is {current_session}")
        
        client_data["awaiting_response"] = True
        client_data["question_time"] = time.time()  # Record when the question finished playing
        
        # Ensure we're using the correct session ID
        if session_id.isdigit():
            client_data["audio_session_id"] = int(session_id)
        
        logger.debug(f"Awaiting response flag set for call {call_id} session {session_id} at {client_data['question_time']}")
        
        # Check if the WebSocket is healthy
        call_sid = client_data.get("callSid")
        if call_sid and call_sid in ACTIVE_STREAMS:
            stream_info = ACTIVE_STREAMS[call_sid]
            if stream_info.get("session_id") != session_id:
                logger.warning(f"Active stream session {stream_info.get('session_id')} doesn't match current session {session_id}")
        
        # Delete the audio file after it has been played to save space
        audio_file_path = os.path.join(AUDIO_DIR, f"response_{call_id}_{session_id}.mp3")
        if os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
            except Exception as e:
                logger.error(f"Error deleting audio file {audio_file_path}: {e}")
        
        # Clean up stale streams
        cleanup_stale_streams()
        
        # Return a TwiML response with a pause to keep the call active
        twiml = VoiceResponse()
        twiml.pause(length=60)  # Adjust length as needed
        return Response(twiml.to_xml(), mimetype='text/xml')
    
    return "Invalid call_id", 400

@app.route('/audio/<call_id>/<session_id>', methods=['GET'])
def get_audio(call_id, session_id):
    """Serve the audio file for a response with session ID"""
    audio_file = os.path.join(AUDIO_DIR, f"response_{call_id}_{session_id}.mp3")
    if os.path.exists(audio_file):
        return send_file(audio_file, mimetype='audio/mpeg')
    
    # Fallback to the old path format if needed
    fallback_file = os.path.join(AUDIO_DIR, f"response_{call_id}.mp3")
    if os.path.exists(fallback_file):
        return send_file(fallback_file, mimetype='audio/mpeg')
        
    return "Audio file not found", 404

@sock.route('/media/<call_id>/<session_id>')
def media(ws, call_id, session_id):
    """Handle WebSocket connection for streaming audio with session isolation and forced closure"""
    # Check if call exists
    if call_id not in connected_clients:
        logger.error(f"Received media for unknown call_id: {call_id}")
        ws.close()
        return
        
    client_data = connected_clients[call_id]
    
    # Validate session ID
    expected_session_id = str(client_data.get("audio_session_id", 0))
    if session_id != expected_session_id:
        logger.warning(f"Received media for session {session_id}, but expected {expected_session_id}")
        # Immediately close this websocket since it's for an outdated session
        logger.debug(f"Closing outdated WebSocket for session {session_id}")
        ws.close()
        return
    
    # Record this connection in active streams
    if client_data.get("callSid"):
        # If there's an existing stream for this call SID, explicitly close it
        if client_data["callSid"] in ACTIVE_STREAMS:
            old_stream = ACTIVE_STREAMS[client_data["callSid"]]
            if "websocket" in old_stream and old_stream["websocket"] != ws:
                try:
                    logger.debug(f"Forcibly closing old WebSocket for session {old_stream.get('session_id', 'unknown')}")
                    old_stream["websocket"].close()
                except Exception as e:
                    logger.error(f"Error closing old WebSocket: {e}")
        
        # Register this as the active stream
        ACTIVE_STREAMS[client_data["callSid"]] = {
            "url": f"wss://{host}/media/{call_id}/{session_id}",
            "session_id": session_id,
            "start_time": time.time(),
            "websocket": ws
        }
    
    logger.debug(f"WebSocket connected for call {call_id} session {session_id}")
    client_data["websocket_connected"] = True
    
    # Track message counts for less verbose logging
    message_count = 0
    outdated_message_count = 0
    
    try:
        while True:
            message = ws.receive()
            if message is None:
                break
            
            message_count += 1
            
            # Always validate that this is still the active session
            if str(client_data.get("audio_session_id")) != session_id:
                outdated_message_count += 1
                # Only log every 100 outdated messages to avoid log spam
                if outdated_message_count == 1 or outdated_message_count % 100 == 0:
                    logger.debug(f"Ignoring outdated messages for session {session_id} (current: {client_data.get('audio_session_id')}) - count: {outdated_message_count}")
                continue
                
            handle_ws_message(call_id, message, session_id)
    except Exception as e:
        logger.error(f"Error in WebSocket for call {call_id} session {session_id}: {e}")
    finally:
        logger.debug(f"WebSocket closed for call {call_id} session {session_id} - processed {message_count} messages ({outdated_message_count} outdated)")
        
        # Clean up active streams
        if client_data.get("callSid") and client_data["callSid"] in ACTIVE_STREAMS:
            if ACTIVE_STREAMS[client_data["callSid"]].get("session_id") == session_id:
                ACTIVE_STREAMS.pop(client_data["callSid"], None)
        
        client_data["websocket_connected"] = False
        ws.close()
    return

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)




