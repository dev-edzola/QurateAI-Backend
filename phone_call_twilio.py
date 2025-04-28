import os
import json
import base64
import audioop
import uuid
import time
import wave
import random
from extensions import sock
from flask import request, send_file, Response, jsonify, Blueprint
from twilio.twiml.voice_response import VoiceResponse, Start, Stop
from twilio.rest import Client
from google.cloud import texttospeech
import logging
logger = logging.getLogger('qurate')

import numpy as np
from collections import OrderedDict
from flask_jwt_extended import jwt_required
from file_management import AUDIO_DIR, TRANSCRIPTION_AUDIO_DIR, cleanup_call_audio_files, delete_stale_files
from db_management import get_db_connection
from ai_utils import llm, parse_for_answers, generate_summary_response, get_next_question, parse_for_form_fields, get_default_form_fields
from common_utils import proper_language_code, transcribe_audio

phone_call_twilio_bp = Blueprint('phone_call_twilio_bp', __name__)


# Tracking variables for stream and session management
ACTIVE_STREAMS = {}  # To track active WebSocket connections by call_id
PROCESSED_SESSIONS = {}  # Format: {call_id: {session_id: timestamp}}

conversation_state = {}

# Twilio credentials and host settings
account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
twilio_client = Client(account_sid, auth_token)
twilio_phone_number = os.environ.get('TWILIO_PHONE_NUMBER')
host = os.environ.get('HOST_DOMAIN')  # e.g., your ngrok or public domain
POP_SOUND_URL = f"https://{host}/static/audio/pop.mp3"





# In-memory storage for call state (keyed by call_id)
connected_clients = {}

# Configuration for silence detection (in seconds)
INITIAL_WAIT_TIMEOUT = 10  # Wait 10 seconds for user to start talking
SILENCE_TIMEOUT = 3        # Wait 3 seconds of silence after user stops talking
MIN_RMS_THRESHOLD = 10     # RMS threshold to detect speech
MIN_AUDIO_BUFFER_LENGTH = 1000  # Minimum audio buffer size to process
MAX_AUDIO_DURATION = 30  # Maximum seconds of audio to record before forcing processing
MAX_AUDIO_BYTES = MAX_AUDIO_DURATION * 8000 * 1  # 8kHz, 1 byte per sample for µ-law

# Initialize Text-to-Speech client
tts_client = texttospeech.TextToSpeechClient()



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
            logger.info(f"Transcription for call {call_id} session {session_id}: '{transcription}'")
            
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
                logger.info(f"Q: {current_message} A: {transcription}")
                
                # Update the current field's parsed answer if we're in asking state
               
                try:
                        parsed_data = parse_for_answers(
                            client_data["collected_answers"],
                            client_data["form_fields"],
                            llm=llm,
                            form_context=client_data.get("form_context", "")
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


def async_play_pop_sound(call_sid):
    # Build a VoiceResponse that plays the pop sound
    response = VoiceResponse()
    response.play(POP_SOUND_URL)
    response.pause(length=60) 
    try:
        # Update the call with the TwiML generated response.
        # This should trigger Twilio to play the pop sound.
        twilio_client.calls(call_sid).update(twiml=response.to_xml())
        logger.info(f"[{call_sid}] Pop sound played asynchronously via media injection")
    except Exception as e:
        logger.error(f"[{call_sid}] Failed to play pop sound asynchronously: {e}")


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
                    #eventlet.spawn_n(async_play_pop_sound, client_data.get("callSid"))
                    logger.debug(f"User did not speak for {INITIAL_WAIT_TIMEOUT} seconds after question. Moving to next question.")
                    #async_play_pop_sound(client_data.get("callSid", ""))
                    process_with_empty_response(call_id, client_data.get("callSid", ""), session_id)
            
            # CASE 2: User has spoken but has been silent for a while
            else:
                silence_duration = audio_recorder.get_silence_duration()
                
                # If silence threshold reached, process the audio
                # Make sure we're not already processing and this session hasn't been processed yet
                if (silence_duration > SILENCE_TIMEOUT and 
                    not client_data.get("processing", False) and
                    not (call_id in PROCESSED_SESSIONS and session_id in PROCESSED_SESSIONS[call_id])):
                    #eventlet.spawn_n(async_play_pop_sound, client_data.get("callSid"))
                    logger.debug(f"Detected {SILENCE_TIMEOUT}s of silence after user spoke. Processing audio.")
                    # async_play_pop_sound(client_data.get("callSid", ""))
                    process_user_audio(call_id, client_data.get("callSid", ""), session_id)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk for call {call_id}: {e}")

# --- HTTP Endpoints ---
@phone_call_twilio_bp.route('/make-call', methods=['POST'])
@jwt_required()
def make_call():
    delete_stale_files(AUDIO_DIR, max_age_seconds=600)
    delete_stale_files(TRANSCRIPTION_AUDIO_DIR, max_age_seconds=600)
    data = {}
    if request.is_json:
        data = request.get_json()
    to_number = request.form.get('to') or data.get('to')
    # user_query = request.form.get('query') or data.get('query')
    form_fields_id = request.form.get("form_fields_id") or data.get("form_fields_id")

    if not to_number:
        return "Please provide a 'to' phone number", 400

    form_fields = []
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            if form_fields_id:
                cursor.execute("""
                    SELECT form_fields, form_context
                    FROM form_fields
                    WHERE id = %s AND is_active = 1
                """, (form_fields_id,))
                result = cursor.fetchone()

                if not result:
                    return jsonify({"error": "Invalid form_fields id"}), 400

                # 2. JSON-decode both
                form_fields  = json.loads(result["form_fields"])
                form_context = result["form_context"] if result["form_context"] is not None else ''
            # elif user_query:
            #     try:
            #         form_fields_data = parse_for_form_fields(user_query, llm)
            #         form_fields = form_fields_data.get('fields', [])
            #         if not form_fields:
            #             raise ValueError("No form fields generated")
            #     except Exception as e:
            #         logger.error(f"Error parsing form fields from query: {e}")
            #         form_fields = get_default_form_fields()
            else:
                form_fields = get_default_form_fields()

            # Prepare call state
            call_id = str(uuid.uuid4())
            connected_clients[call_id] = {
                "audio_recorder": AudioRecorder(),
                "current_state": "initial",
                "callSid": None,
                "last_voice": time.time(),
                "processing": False,
                "last_processed": 0,
                "form_fields": form_fields,
                "collected_answers": OrderedDict(),
                "field_parsed_answers": {field["field_id"]: None for field in form_fields},
                "field_asked_counter": {field["field_id"]: 0 for field in form_fields},
                "language": "en-IN",
                "greeting_message": "Hello! I'm Meera, your AI assistant. It's lovely to connect with you.",
                "awaiting_response": False,
                "question_time": 0,
                "max_wait_time": INITIAL_WAIT_TIMEOUT,
                "silence_timeout": SILENCE_TIMEOUT,
                "audio_session_id": 0,
                "form_context": form_context,
            }

            logger.info(f"Generated form fields for call {call_id}: {json.dumps(form_fields)}")
            call = twilio_client.calls.create(
            to=to_number,
            from_=twilio_phone_number,
            url=f"https://{host}/voice?call_id={call_id}"
        )
            
            # Insert communication into DB
            cursor.execute("""
                INSERT INTO communications 
                (communication_type, form_fields_id, collected_answers, field_asked_counter, 
                 language_info, field_parsed_answers, call_id, call_sid, communication_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                "phone_call",
                form_fields_id,
                json.dumps(connected_clients[call_id]["collected_answers"]),
                json.dumps(connected_clients[call_id]["field_asked_counter"]),
                connected_clients[call_id]["language"],
                json.dumps(connected_clients[call_id]["field_parsed_answers"]),
                call_id,
                call.sid,
                'Not Started'
            ))
            connection.commit()
            connected_clients[call_id]["communication_id"] = cursor.lastrowid
            

    except Exception as twilio_error:
        logger.error(f"Twilio call initiation failed: {twilio_error}")
        return jsonify({"error": "Failed to initiate call"}), 500        
    finally:
        connection.close()

    logger.info(f"Call initiated to {to_number} with call_id: {call_id}")
    return jsonify({
        "message": f"Call initiated with SID: {call.sid}",
        "call_id": call_id,
        "form_fields": form_fields
    }), 200


@phone_call_twilio_bp.route('/voice', methods=['POST'])
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
            call_id=call_id,
            audio=True,
            form_context=client_data.get("form_context")
        )
        communication_status = 'In Progress'
        if not client_data["collected_answers"]:
            communication_status = 'Not Started'
        if field_id is None:
            client_data["current_state"] = "complete"
            communication_status = 'Completed'
        else:
            client_data["current_field"] = field_id
            client_data["current_state"] = "asking"
            client_data["field_asked_counter"][field_id] += 1
            client_data["current_message"] = message

        if(client_data.get("communication_id")):
            
            connection = get_db_connection()
            try:
                with connection.cursor() as cursor:
                    update_sql = """
                        UPDATE communications 
                        SET collected_answers = %s, field_asked_counter = %s, field_parsed_answers = %s, language_info = %s, communication_status = %s 
                        WHERE communication_id = %s
                    """
                    cursor.execute(update_sql, (
                        json.dumps(client_data["collected_answers"]),
                        json.dumps(client_data["field_asked_counter"]),
                        json.dumps(client_data["field_parsed_answers"]),
                        client_data.get("language"),
                        communication_status,
                        client_data.get("communication_id")
                    ))
                    connection.commit()
            except Exception as e:
                connection.rollback()
                return jsonify({"error": str(e)}), 500
            finally:
                connection.close()

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
        logger.info(f"Hanging up call {call_id} after final message")
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
    
    logger.info(f"Question asked: {message}")
    return Response(response.to_xml(), mimetype='text/xml')

@phone_call_twilio_bp.route('/set-awaiting-response', methods=['GET', 'POST'])
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

@phone_call_twilio_bp.route('/audio/<call_id>/<session_id>', methods=['GET'])
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
