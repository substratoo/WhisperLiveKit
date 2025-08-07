#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging
from .backends import FasterWhisperASR, MLXWhisper, WhisperTimestampedASR, OpenaiApiASR, SimulStreamingASR, SIMULSTREAMING_AVAILABLE, SIMULSTREAMING_ERROR_AND_INSTALLATION_INSTRUCTIONS
from .online_asr import OnlineASRProcessor, VACOnlineASRProcessor, SimulStreamingOnlineProcessor, SIMULSTREAMING_AVAILABLE as SIMULSTREAMING_ONLINE_AVAILABLE

logger = logging.getLogger(__name__)



WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(
    ","
)


def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert (
        lan in WHISPER_LANG_CODES
    ), "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk

        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)

        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if (
        lan
        in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split()
    ):
        from mosestokenizer import MosesSentenceSplitter        

        return MosesSentenceSplitter(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if (
        lan
        in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split()
    ):
        logger.debug(
            f"{lan} code is not supported by wtpsplit. Going to use None lang_code option."
        )
        lan = None

    from wtpsplit import WtP

    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")

    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)

    return WtPtok()


def backend_factory(args):
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=args.lan)
    elif backend == "simulstreaming":
        logger.debug("Using SimulStreaming backend.")
        if not SIMULSTREAMING_AVAILABLE:
            raise SIMULSTREAMING_ERROR_AND_INSTALLATION_INSTRUCTIONS
        
        simulstreaming_kwargs = {}
        for attr in ['frame_threshold', 'beams', 'decoder_type', 'audio_max_len', 'audio_min_len', 
                     'cif_ckpt_path', 'never_fire', 'init_prompt', 'static_init_prompt', 
                     'max_context_tokens', 'model_path']:
            if hasattr(args, attr):
                simulstreaming_kwargs[attr] = getattr(args, attr)
        
        # Add segment_length from min_chunk_size
        simulstreaming_kwargs['segment_length'] = getattr(args, 'min_chunk_size', 0.5)
        simulstreaming_kwargs['task'] = args.task
        
        size = args.model
        t = time.time()
        logger.info(f"Loading SimulStreaming {size} model for language {args.lan}...")
        asr = SimulStreamingASR(
            modelsize=size,
            lan=args.lan,
            cache_dir=getattr(args, 'model_cache_dir', None),
            model_dir=getattr(args, 'model_dir', None),
            **simulstreaming_kwargs
        )
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        elif backend == "mlx-whisper":
            asr_cls = MLXWhisper
        else:
            asr_cls = WhisperTimestampedASR

        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model with language={args.lan}, device={args.device} and compute_type={args.compute_type}...")
        asr = asr_cls(
            modelsize=size,
            lan=args.lan,
            cache_dir=getattr(args, 'model_cache_dir', None),
            model_dir=getattr(args, 'model_dir', None),
            device=getattr(args, 'device', 'auto'),
            compute_type=getattr(args, 'compute_type', 'auto')
        )
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Apply common configurations
    if getattr(args, "vad", False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan
    if args.task == "translate":
        if backend != "simulstreaming":
            asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None
    return asr, tokenizer

def online_factory(args, asr, tokenizer, logfile=sys.stderr):
    if args.backend == "simulstreaming":
        if not SIMULSTREAMING_ONLINE_AVAILABLE:
            raise SIMULSTREAMING_ERROR_AND_INSTALLATION_INSTRUCTIONS
        
        logger.debug("Creating SimulStreaming online processor")
        online = SimulStreamingOnlineProcessor(
            asr,
            tokenizer,
            logfile=logfile,
            buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec),
            confidence_validation=args.confidence_validation
        )
    elif args.vac:
        online = VACOnlineASRProcessor(
            args.min_chunk_size,
            asr,
            tokenizer,
            logfile=logfile,
            buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec),
            confidence_validation = args.confidence_validation
        )
    else:
        online = OnlineASRProcessor(
            asr,
            tokenizer,
            logfile=logfile,
            buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec),
            confidence_validation = args.confidence_validation
        )
    return online
  
def asr_factory(args, logfile=sys.stderr):
    """
    Creates and configures an ASR and ASR Online instance based on the specified backend and arguments.
    """
    asr, tokenizer = backend_factory(args)
    online = online_factory(args, asr, tokenizer, logfile=logfile)
    return asr, online

def warmup_asr(asr, warmup_file=None, timeout=5):
    """
    Warmup the ASR model by transcribing a short audio file.
    """
    import os
    import tempfile
    
    is_simulstreaming = hasattr(asr, 'warmup') and callable(getattr(asr, 'warmup'))
    
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
    
    print(f"Warming up {'SimulStreaming' if is_simulstreaming else 'Whisper'} with {warmup_file}")
    try:
        import librosa
        audio, sr = librosa.load(warmup_file, sr=16000)
    except Exception as e:
        logger.warning(f"Failed to load audio file: {e}")
        return False
    
    try:
        if is_simulstreaming:
            asr.warmup(audio)
        else:
            asr.transcribe(audio)
        
        logger.info(f"{'SimulStreaming' if is_simulstreaming else 'Whisper'} is warmed up")
        return True
        
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
        return False
