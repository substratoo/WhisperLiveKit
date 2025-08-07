import gc
import sys
import logging
import io
import soundfile as sf
import math
try: 
    import torch
except ImportError: 
    torch = None
from typing import List
import numpy as np
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.simul_whisper.license_simulstreaming import SIMULSTREAMING_LICENSE
logger = logging.getLogger(__name__)
SIMULSTREAMING_ERROR_AND_INSTALLATION_INSTRUCTIONS = ImportError(
"""SimulStreaming dependencies are not available.
Please install WhisperLiveKit using pip install "whisperlivekit[simulstreaming]"
""")

try:
    from whisperlivekit.simul_whisper.config import AlignAttConfig
    from whisperlivekit.simul_whisper.simul_whisper import PaddedAlignAttWhisper, DEC_PAD
    from whisperlivekit.simul_whisper.whisper import tokenizer
    SIMULSTREAMING_AVAILABLE = True
except ImportError:
    SIMULSTREAMING_AVAILABLE = False
    AlignAttConfig = None
    PaddedAlignAttWhisper = None
    DEC_PAD = None
    tokenizer = None

class ASRBase:
    sep = " "  # join transcribe words with this character (" " for whisper_timestamped,
              # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, device="auto", compute_type="auto", logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(modelsize, cache_dir, model_dir, device, compute_type)

    def with_offset(self, offset: float) -> ASRToken:
        # This method is kept for compatibility (typically you will use ASRToken.with_offset)
        return ASRToken(self.start + offset, self.end + offset, self.text)

    def __repr__(self):
        return f"ASRToken(start={self.start:.2f}, end={self.end:.2f}, text={self.text!r})"

    def load_model(self, modelsize, cache_dir, model_dir, device="auto", compute_type="auto"):
        raise NotImplementedError("must be implemented in the child class")
    
    def unload_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def transcribe(self, audio, init_prompt=""):
        raise NotImplementedError("must be implemented in the child class")

    def use_vad(self):
        raise NotImplementedError("must be implemented in the child class")


class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped as the backend."""
    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None, device="auto", compute_type="auto"):
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped

        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir) # TODO: use device and compute_type

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            verbose=None,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return result

    def ts_words(self, r) -> List[ASRToken]:
        """
        Converts the whisper_timestamped result to a list of ASRToken objects.
        """
        tokens = []
        for segment in r["segments"]:
            for word in segment["words"]:
                token = ASRToken(word["start"], word["end"], word["text"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [segment["end"] for segment in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper as the backend."""
    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None, device="auto", compute_type="auto"):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. "
                         f"modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("Either modelsize or model_dir must be set")

        model = WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> list:
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                token = ASRToken(word.start, word.end, word.word, probability=word.probability)
                tokens.append(token)
        return tokens

    def segments_end_ts(self, segments) -> List[float]:
        return [segment.end for segment in segments]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper optimized for Apple Silicon.
    """
    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
            logger.debug(f"Loading whisper model {modelsize}. You use mlx whisper, so {model_size_or_path} will be used.")
        else:
            raise ValueError("Either modelsize or model_dir must be set")

        self.model_size_or_path = model_size_or_path
        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }
        mlx_model_path = model_mapping.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        if self.transcribe_kargs:
            logger.warning("Transcribe kwargs (vad, task) are not compatible with MLX Whisper and will be ignored.")
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
        )
        return segments.get("segments", [])

    def ts_words(self, segments) -> List[ASRToken]:
        tokens = []
        for segment in segments:
            if segment.get("no_speech_prob", 0) > 0.9:
                continue
            for word in segment.get("words", []):
                token = ASRToken(word["start"], word["end"], word["word"], probability=word["probability"])
                tokens.append(token)
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for transcription."""
    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature
        self.load_model()
        self.use_vad_opt = False
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments) -> List[ASRToken]:
        """
        Converts OpenAI API response words into ASRToken objects while
        optionally skipping words that fall into no-speech segments.
        """
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment.no_speech_prob > 0.8:
                    no_speech_segments.append((segment.start, segment.end))
        tokens = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            tokens.append(ASRToken(start, end, word.word))
        return tokens

    def segments_end_ts(self, res) -> List[float]:
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)
        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt
        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"


class SimulStreamingASR(ASRBase):
    """SimulStreaming backend with AlignAtt policy."""
    sep = ""

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr, **kwargs):
        if not SIMULSTREAMING_AVAILABLE:
            raise SIMULSTREAMING_ERROR_AND_INSTALLATION_INSTRUCTIONS
        logger.warning(SIMULSTREAMING_LICENSE)
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        
        self.model_path = kwargs.get('model_path', './large-v3.pt')
        self.frame_threshold = kwargs.get('frame_threshold', 25)
        self.audio_max_len = kwargs.get('audio_max_len', 30.0)
        self.audio_min_len = kwargs.get('audio_min_len', 0.0)
        self.segment_length = kwargs.get('segment_length', 0.5)
        self.beams = kwargs.get('beams', 1)
        self.decoder_type = kwargs.get('decoder_type', 'greedy' if self.beams == 1 else 'beam')
        self.task = kwargs.get('task', 'transcribe')
        self.cif_ckpt_path = kwargs.get('cif_ckpt_path', None)
        self.never_fire = kwargs.get('never_fire', False)
        self.init_prompt = kwargs.get('init_prompt', None)
        self.static_init_prompt = kwargs.get('static_init_prompt', None)
        self.max_context_tokens = kwargs.get('max_context_tokens', None)
        
        if model_dir is not None:
            self.model_path = model_dir
        elif modelsize is not None: #For the moment the .en.pt models do not work!
            model_mapping = {
                'tiny': './tiny.pt',
                'base': './base.pt',
                'small': './small.pt',
                'medium': './medium.pt',
                'medium.en': './medium.en.pt',
                'large-v1': './large-v1.pt',
                'base.en': './base.en.pt',
                'small.en': './small.en.pt',
                'tiny.en': './tiny.en.pt',
                'large-v2': './large-v2.pt',
                'large-v3': './large-v3.pt',
                'large': './large-v3.pt'
            }
            self.model_path = model_mapping.get(modelsize, f'./{modelsize}.pt')
        
        self.model = self.load_model(modelsize, cache_dir, model_dir)
        
        # Set up tokenizer for translation if needed
        if self.task == "translate":
            self.set_translate_task()

    def load_model(self, modelsize, cache_dir, model_dir):
        try:
            cfg = AlignAttConfig(
                model_path=self.model_path,
                segment_length=self.segment_length,
                frame_threshold=self.frame_threshold,
                language=self.original_language,
                audio_max_len=self.audio_max_len,
                audio_min_len=self.audio_min_len,
                cif_ckpt_path=self.cif_ckpt_path,
                decoder_type="beam",
                beam_size=self.beams,
                task=self.task,
                never_fire=self.never_fire,
                init_prompt=self.init_prompt,
                max_context_tokens=self.max_context_tokens,
                static_init_prompt=self.static_init_prompt,
            )
            
            logger.info(f"Loading SimulStreaming model with language: {self.original_language}")
            model = PaddedAlignAttWhisper(cfg)
            return model
            
        except Exception as e:
            logger.error(f"Failed to load SimulStreaming model: {e}")
            raise

    def transcribe(self, audio, init_prompt=""):
        """Transcribe audio using SimulStreaming."""
        try:
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio
                
            prompt = init_prompt if init_prompt else (self.init_prompt or "")
            
            result = self.model.infer(audio_tensor, init_prompt=prompt)
            
            if torch.is_tensor(result):
                result = result[result < DEC_PAD]
                
            logger.debug(f"SimulStreaming transcription result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"SimulStreaming transcription failed: {e}")
            raise

    def ts_words(self, result) -> List[ASRToken]:
        """Convert SimulStreaming result to ASRToken list."""
        tokens = []
        
        try:
            if torch.is_tensor(result):
                text = self.model.tokenizer.decode(result.cpu().numpy())
            else:
                text = str(result)
                
            if not text or len(text.strip()) == 0:
                return tokens
                
            # We dont have word-level timestamps here. 1rst approach, should be improved later.
            words = text.strip().split()
            if not words:
                return tokens
            
            duration_per_word = 0.1  # this will be modified based on actual audio duration 
            #with the SimulStreamingOnlineProcessor
            
            for i, word in enumerate(words):
                start_time = i * duration_per_word
                end_time = (i + 1) * duration_per_word
                
                token = ASRToken(
                    start=start_time,
                    end=end_time,
                    text=word,
                    probability=1.0
                )
                tokens.append(token)
                
        except Exception as e:
            logger.error(f"Error converting SimulStreaming result to tokens: {e}")
            
        return tokens

    def segments_end_ts(self, result) -> List[float]:
        """Get segment end timestamps."""
        if torch.is_tensor(result):
            num_tokens = len(result)
            return [num_tokens * 0.1]  # rough estimate
        return [1.0]

    def use_vad(self):
        """Enable VAD - SimulStreaming has different VAD handling."""
        logger.info("VAD requested for SimulStreaming - handled internally by the model")
        pass

    def set_translate_task(self):
        """Set up translation task."""
        try:
            self.model.tokenizer = tokenizer.get_tokenizer(
                multilingual=True,
                language=self.model.cfg.language,
                num_languages=self.model.model.num_languages,
                task="translate"
            )
            logger.info("SimulStreaming configured for translation task")
        except Exception as e:
            logger.error(f"Failed to configure SimulStreaming for translation: {e}")
            raise

    def warmup(self, audio, init_prompt=""):
        """Warmup the SimulStreaming model."""
        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            self.model.insert_audio(audio)
            self.model.infer(True)
            self.model.refresh_segment(complete=True)
            logger.info("SimulStreaming model warmed up successfully")
        except Exception as e:
            logger.exception(f"SimulStreaming warmup failed: {e}")
