from google.cloud import speech
import logging
import json
from datetime import datetime

logger = logging.getLogger('qurate')
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
            logger.info(f"Speech recognized with {language_code}: '{transcription.strip()}' (Avg confidence: {avg_confidence:.2f})")
        
        return transcription.strip()
    except Exception as e:
        logger.error(f"Speech recognition error with language {language_code}: {e}")
        return ""

def _safe_json_load(val):
    """If val is JSON‑encoded text, parse it; otherwise return as is."""
    if not val:
        return None
    try:
        return json.loads(val)
    except (ValueError, TypeError):
        return val

def _format_datetime(dt):
    """Convert datetime to ISO string or return as‑is."""
    if isinstance(dt, datetime):
        return dt.isoformat()
    return dt
