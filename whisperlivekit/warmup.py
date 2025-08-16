
import logging

logger = logging.getLogger(__name__)

def load_file(warmup_file=None, timeout=5):
    import os
    import tempfile
    import librosa
        
    if warmup_file is None:
        # Download JFK sample if not already present
        jfk_url = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
        temp_dir = tempfile.gettempdir()
        warmup_file = os.path.join(temp_dir, "whisper_warmup_jfk.wav")
        
        if not os.path.exists(warmup_file):
            logger.debug(f"Downloading warmup file from {jfk_url}")
            print(f"Downloading warmup file from {jfk_url}")
            import time
            import urllib.request
            import urllib.error
            import socket
            
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(timeout)
            
            start_time = time.time()
            try:
                urllib.request.urlretrieve(jfk_url, warmup_file)
                logger.debug(f"Download successful in {time.time() - start_time:.2f}s")
            except (urllib.error.URLError, socket.timeout) as e:
                logger.warning(f"Download failed: {e}. Proceeding without warmup.")
                return False
            finally:
                socket.setdefaulttimeout(original_timeout)
    elif not warmup_file:
        return False 
    
    if not warmup_file or not os.path.exists(warmup_file) or os.path.getsize(warmup_file) == 0:
        logger.warning(f"Warmup file {warmup_file} invalid or missing.")
        return False
    
    try:
        audio, sr = librosa.load(warmup_file, sr=16000)
    except Exception as e:
        logger.warning(f"Failed to load audio file: {e}")
        return False
    return audio

def warmup_asr(asr, warmup_file=None, timeout=5):
    """
    Warmup the ASR model by transcribing a short audio file.
    """
    audio = load_file(warmup_file=None, timeout=5)
    asr.transcribe(audio)
    logger.info("ASR model is warmed up")
    
def warmup_online(online, warmup_file=None, timeout=5):
    audio = load_file(warmup_file=None, timeout=5)
    online.warmup(audio)
    logger.warning("ASR is warmed up")