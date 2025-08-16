try:
    from whisperlivekit.whisper_streaming_custom.whisper_online import backend_factory
    from whisperlivekit.whisper_streaming_custom.online_asr import VACOnlineASRProcessor, OnlineASRProcessor
except ImportError:
    from .whisper_streaming_custom.whisper_online import backend_factory
    from .whisper_streaming_custom.online_asr import VACOnlineASRProcessor, OnlineASRProcessor
from whisperlivekit.warmup import warmup_asr, warmup_online
from argparse import Namespace
import gc
import logging
import torch

logger = logging.getLogger(__name__)
import sys

class TranscriptionEngine:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, **kwargs):
        if TranscriptionEngine._initialized:
            return

        defaults = {
            "host": "localhost",
            "port": 8000,
            "warmup_file": None,
            "diarization": False,
            "punctuation_split": False,
            "min_chunk_size": 0.5,
            "model": "tiny",
            "model_cache_dir": None,
            "model_dir": None,
            "lan": "auto",
            "task": "transcribe",
            "backend": "faster-whisper",
            "vac": False,
            "vac_chunk_size": 0.04,
            "log_level": "DEBUG",
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "transcription": True,
            "vad": True,
            "device": "auto",
            "compute_type": "auto",
            # whisperstreaming params:
            "buffer_trimming": "segment",
            "confidence_validation": False,
            "buffer_trimming_sec": 15,
            # simulstreaming params:
            "frame_threshold": 25,
            "beams": 1,
            "decoder_type": None,
            "audio_max_len": 30.0,
            "audio_min_len": 0.0,
            "cif_ckpt_path": None,
            "never_fire": False,
            "init_prompt": None,
            "static_init_prompt": None,
            "max_context_tokens": None,
            "model_path": './base.pt',
            # diart params:
            "segmentation_model": "pyannote/segmentation-3.0",
            "embedding_model": "pyannote/embedding",

        }

        config_dict = {**defaults, **kwargs}

        if 'no_transcription' in kwargs:
            config_dict['transcription'] = not kwargs['no_transcription']
        if 'no_vad' in kwargs:
            config_dict['vad'] = not kwargs['no_vad']
        
        config_dict.pop('no_transcription', None)
        config_dict.pop('no_vad', None)

        if 'language' in kwargs:
            config_dict['lan'] = kwargs['language']
        config_dict.pop('language', None) 

        self.args = Namespace(**config_dict)
        
        self.asr = None
        self.tokenizer = None
        self.diarization = None
        
        if self.args.transcription:
            if self.args.backend == "simulstreaming": 
                from simul_whisper import SimulStreamingASR
                self.tokenizer = None
                simulstreaming_kwargs = {}
                for attr in ['frame_threshold', 'beams', 'decoder_type', 'audio_max_len', 'audio_min_len', 
                            'cif_ckpt_path', 'never_fire', 'init_prompt', 'static_init_prompt', 
                            'max_context_tokens', 'model_path', 'warmup_file', 'preload_model_count']:
                    if hasattr(self.args, attr):
                        simulstreaming_kwargs[attr] = getattr(self.args, attr)
        
                # Add segment_length from min_chunk_size
                simulstreaming_kwargs['segment_length'] = getattr(self.args, 'min_chunk_size', 0.5)
                simulstreaming_kwargs['task'] = self.args.task
                
                size = self.args.model
                self.asr = SimulStreamingASR(
                    modelsize=size,
                    lan=self.args.lan,
                    cache_dir=getattr(self.args, 'model_cache_dir', None),
                    model_dir=getattr(self.args, 'model_dir', None),
                    **simulstreaming_kwargs
                )

            else:
                self.asr, self.tokenizer = backend_factory(self.args)
            warmup_asr(self.asr, self.args.warmup_file) #for simulstreaming, warmup should be done in the online class not here

        if self.args.diarization:
            from whisperlivekit.diarization.diarization_online import DiartDiarization
            self.diarization = DiartDiarization(
                block_duration=self.args.min_chunk_size,
                segmentation_model_name=self.args.segmentation_model,
                embedding_model_name=self.args.embedding_model
            )
            
        TranscriptionEngine._initialized = True
    
    def free(self):
        # TODO: proper cleanup
        """
        try:
            # self.asr.model = None
            # self.asr = None
            # self.tokenizer = None
            # self.diarization = None
        except AttributeError:
            logger.warning("Skipping free: no model loaded")
            return
        """
        gc.collect()
        torch.cuda.empty_cache()



def online_factory(args, asr, tokenizer, logfile=sys.stderr):
    if args.backend == "simulstreaming":    
        from simul_whisper import SimulStreamingOnlineProcessor
        online = SimulStreamingOnlineProcessor(
            asr,
            logfile=logfile,
        )
        # warmup_online(online, args.warmup_file)
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
  
