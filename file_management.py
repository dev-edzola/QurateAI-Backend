import os
import logging
import glob
import time
from logging.handlers import RotatingFileHandler

# module‑level placeholders; will be set in init
AUDIO_DIR = None
TRANSCRIPTION_AUDIO_DIR = None
LOG_DIR = None

# These loggers are declared now but not yet configured
logger = logging.getLogger('')             # root logger
app_logger = logging.getLogger('qurate')   # your “qurate” logger


def init_file_management():
    """Call this once at startup to set up directories and logging."""
    global AUDIO_DIR, TRANSCRIPTION_AUDIO_DIR, LOG_DIR, logger, app_logger

    # 1) Paths
    file_disk = os.environ.get("File_Disk", "Files").lstrip("/")
    base_dir = os.path.join(os.path.dirname(__file__), file_disk)

    AUDIO_DIR = os.path.join(base_dir, "audio_files")
    os.makedirs(AUDIO_DIR, exist_ok=True)

    TRANSCRIPTION_AUDIO_DIR = os.path.join(base_dir, "transcription_audio")
    os.makedirs(TRANSCRIPTION_AUDIO_DIR, exist_ok=True)

    LOG_DIR = os.path.join(base_dir, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    # 2) File logging (rotating file handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                os.path.join(LOG_DIR, "qurate.log"),
                maxBytes=10 * 1024 * 1024,
                backupCount=5
            )
        ]
    )

    # 3) Console handler (INFO+ only)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # 4) Configure your app_logger
    app_logger.setLevel(logging.INFO)


def cleanup_call_audio_files(call_id):
    """Delete all audio files associated with a specific call_id"""
    logger.info(f"Cleaning up audio files for call {call_id}")
    
    files_deleted = 0
    
    # 1. Clean up TTS response files
    response_pattern = os.path.join(AUDIO_DIR, f"response_{call_id}_*.mp3")
    for file_path in glob.glob(response_pattern):
        try:
            os.remove(file_path)
            logger.info(f"Deleted TTS file: {file_path}")
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
                logger.info(f"Deleted transcription file: {file_path}")
                files_deleted += 1
            except Exception as e:
                logger.error(f"Error deleting transcription file {file_path}: {e}")
    
    logger.info(f"Deleted {files_deleted} files for call {call_id}")
    return files_deleted


def delete_stale_files(directory, max_age_seconds=1200):
    """Delete files older than max_age_seconds in a directory"""
    now = time.time()
    deleted_files = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    logger.info(f"Auto-cleanup: Deleted stale file: {file_path}")
                    deleted_files += 1
                except Exception as e:
                    logger.error(f"Error deleting stale file {file_path}: {e}")
    return deleted_files