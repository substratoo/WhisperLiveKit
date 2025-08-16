import sys
import numpy as np
import logging
from typing import List, Tuple, Optional
import logging
from whisperlivekit.timed_objects import ASRToken, Transcript
from whisperlivekit.warmup import load_file
from whisperlivekit.simul_whisper.license_simulstreaming import SIMULSTREAMING_LICENSE
from .whisper import load_model, tokenizer
import os
import gc
logger = logging.getLogger(__name__)

try:
    import torch
    from whisperlivekit.simul_whisper.config import AlignAttConfig
    from whisperlivekit.simul_whisper.simul_whisper import PaddedAlignAttWhisper
    from whisperlivekit.simul_whisper.whisper import tokenizer
except ImportError as e:
    raise ImportError(
        """SimulStreaming dependencies are not available.
        Please install WhisperLiveKit using pip install "whisperlivekit[simulstreaming]".""")

class SimulStreamingOnlineProcessor:
    SAMPLING_RATE = 16000

    def __init__(
        self,
        asr,
        logfile=sys.stderr,
        warmup_file=None
    ):        
        self.asr = asr
        self.logfile = logfile
        self.is_last = False
        self.beg = 0.0
        self.end = 0.0
        self.cumulative_audio_duration = 0.0
        
        self.committed: List[ASRToken] = []
        self.last_result_tokens: List[ASRToken] = []
        model = asr.get_new_model_instance()
        self.model = PaddedAlignAttWhisper(
            cfg=asr.cfg,
            loaded_model=model)
        if asr.tokenizer:
            self.model.tokenizer = asr.tokenizer

    def insert_audio_chunk(self, audio: np.ndarray, audio_stream_end_time: Optional[float] = None):
        """Append an audio chunk to be processed by SimulStreaming."""
            
        # Convert numpy array to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Update timing
        chunk_duration = len(audio) / self.SAMPLING_RATE
        self.cumulative_audio_duration += chunk_duration
        
        if audio_stream_end_time is not None:
            self.end = audio_stream_end_time
        else:
            self.end = self.cumulative_audio_duration            
        self.model.insert_audio(audio_tensor)

    def get_buffer(self):
        return Transcript(
            start=None, 
            end=None, 
            text='', 
            probability=None
        )

    def timestamped_text(self, tokens, generation):
        # From the simulstreaming repo. self.model to self.asr.model
        pr = generation["progress"]
        if "result" not in generation:
            split_words, split_tokens = self.model.tokenizer.split_to_word_tokens(tokens)
        else:
            split_words, split_tokens = generation["result"]["split_words"], generation["result"]["split_tokens"]

        frames = [p["most_attended_frames"][0] for p in pr]
        tokens = tokens.copy()
        ret = []
        for sw,st in zip(split_words,split_tokens):
            b = None
            for stt in st:
                t,f = tokens.pop(0), frames.pop(0)
                if t != stt:
                    raise ValueError(f"Token mismatch: {t} != {stt} at frame {f}.")
                if b is None:
                    b = f
            e = f
            out = (b*0.02, e*0.02, sw)
            ret.append(out)
            logger.debug(f"TS-WORD:\t{' '.join(map(str, out))}")
        return ret

    def process_iter(self) -> Tuple[List[ASRToken], float]:
        """
        Process accumulated audio chunks using SimulStreaming.
        
        Returns a tuple: (list of committed ASRToken objects, float representing the audio processed up to time).
        """
        try:            
            tokens, generation_progress = self.model.infer(is_last=self.is_last)
            ts_words = self.timestamped_text(tokens, generation_progress)
            
            new_tokens = []
            for ts_word in ts_words:
                
                start, end, word = ts_word
                token = ASRToken(
                    start=start,
                    end=end,
                    text=word,
                    probability=0.95  # fake prob. Maybe we can extract it from the model?
                )
                new_tokens.append(token)
                self.committed.extend(new_tokens)
            
            return new_tokens, self.end

            
        except Exception as e:
            logger.exception(f"SimulStreaming processing error: {e}")
            return [], self.end

    def warmup(self, audio, init_prompt=""):
        """Warmup the SimulStreaming model."""
        try:
            self.model.insert_audio(audio)
            self.model.infer(True)
            self.model.refresh_segment(complete=True)
            logger.info("SimulStreaming model warmed up successfully")
        except Exception as e:
            logger.exception(f"SimulStreaming warmup failed: {e}")

    def __del__(self):
        # free the model and add a new model to stack.
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.asr.new_model_to_stack()

class SimulStreamingASR():
    """SimulStreaming backend with AlignAtt policy."""
    sep = ""

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr, **kwargs):
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
        self.warmup_file = kwargs.get('warmup_file', None)
        self.preload_model_count = kwargs.get('preload_model_count', 1)
        
        if model_dir is not None:
            self.model_path = model_dir
        elif modelsize is not None:
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
        
        # Set up tokenizer for translation if needed
        if self.task == "translate":
            self.tokenizer = self.set_translate_task()
        else:
            self.tokenizer = None
        self.cfg = AlignAttConfig(
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
        
        self.model_name = os.path.basename(self.cfg.model_path).replace(".pt", "")
        self.model_path = os.path.dirname(os.path.abspath(self.cfg.model_path))
        self.models = [self.load_model() for i in range(self.preload_model_count)]
    



    def load_model(self):
        whisper_model = load_model(name=self.model_name, download_root=self.model_path)
        warmup_audio = load_file(self.warmup_file)
        whisper_model.transcribe(warmup_audio, language=self.original_language)
        return whisper_model
    
    def get_new_model_instance(self):
        """
        SimulStreaming cannot share the same backend because it uses global forward hooks on the attention layers.
        Therefore, each user requires a separate model instance, which can be memory-intensive. To maintain speed, we preload the models into memory.
        """
        if len(self.models) == 0:
            self.models.append(self.load_model())
        new_model = self.models.pop()
        return new_model

    def new_model_to_stack(self):
        self.models.append(self.load_model())
        

    def set_translate_task(self):
        """Set up translation task."""
        return tokenizer.get_tokenizer(
            multilingual=True,
            language=self.model.cfg.language,
            num_languages=self.model.model.num_languages,
            task="translate"
        )

    def transcribe(self, audio):
        """
        Warmup is done directly in load_model
        """
        pass