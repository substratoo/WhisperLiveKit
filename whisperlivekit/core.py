try:
    from whisperlivekit.whisper_streaming_custom.whisper_online import backend_factory, warmup_asr
except ImportError:
    from .whisper_streaming_custom.whisper_online import backend_factory, warmup_asr
from argparse import Namespace


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
            "confidence_validation": False,
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
            "buffer_trimming": "segment",
            "buffer_trimming_sec": 15,
            "log_level": "DEBUG",
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "transcription": True,
            "vad": True,
            "segmentation_model": "pyannote/segmentation-3.0",
            "embedding_model": "pyannote/embedding",
            "device": "auto",
            "compute_type": "auto",
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
            self.asr, self.tokenizer = backend_factory(self.args)
            warmup_asr(self.asr, self.args.warmup_file)

        if self.args.diarization:
            from whisperlivekit.diarization.diarization_online import DiartDiarization
            self.diarization = DiartDiarization(
                block_duration=self.args.min_chunk_size,
                segmentation_model_name=self.args.segmentation_model,
                embedding_model_name=self.args.embedding_model
            )
            
        TranscriptionEngine._initialized = True
