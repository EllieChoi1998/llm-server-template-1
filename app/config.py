from pydantic import BaseSettings
from typing import List, Dict, Optional, Any, Literal
import os
from pathlib import Path


class Settings(BaseSettings):
    # 기본 설정
    APP_NAME: str = "Local LLM API"
    API_VERSION: str = "v1"
    DEBUG: bool = False
    
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # CORS 설정
    CORS_ORIGINS: List[str] = ["*"]
    
    # 모델 유형
    MODEL_BACKEND: Literal["llama-cpp", "transformers", "vllm", "auto"] = os.environ.get("MODEL_BACKEND", "auto")
    
    # GGUF 모델 설정
    MODEL_DIR: str = os.environ.get("MODEL_DIR", "./models")
    DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "llama3-8b-instruct.Q4_K_M.gguf")
    AVAILABLE_MODELS: Dict[str, str] = {
        "llama3-8b": "llama3-8b-instruct.Q4_K_M.gguf",
        "llama3-70b": "llama3-70b-instruct.Q4_K_M.gguf",
        "mistral-7b": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "gemma-7b": "gemma-7b-it.Q4_K_M.gguf"
    }
    
    # Transformers 모델 설정
    TRANSFORMERS_DEFAULT_MODEL: str = os.environ.get("TRANSFORMERS_DEFAULT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    TRANSFORMERS_MODELS: Dict[str, Dict[str, Any]] = {
        "llama3-8b": {
            "model_id": "meta-llama/Llama-3-8B-Instruct",
            "load_in_8bit": True
        },
        "gemma-7b": {
            "model_id": "google/gemma-7b-it",
            "load_in_8bit": True
        },
        "mistral-7b": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "load_in_8bit": True
        },
        "tinyllama": {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "load_in_8bit": False
        }
    }
    
    # vLLM 모델 설정
    VLLM_DEFAULT_MODEL: str = os.environ.get("VLLM_DEFAULT_MODEL", "meta-llama/Llama-3-8B-Instruct")
    VLLM_MODELS: Dict[str, Dict[str, Any]] = {
        "llama3-8b": {
            "model_id": "meta-llama/Llama-3-8B-Instruct",
            "tensor_parallel_size": 1
        },
        "llama3-70b": {
            "model_id": "meta-llama/Llama-3-70B-Instruct",
            "tensor_parallel_size": 2,
            "swap_space": 4
        },
        "mistral-7b": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "tensor_parallel_size": 1
        },
        "phi3-mini": {
            "model_id": "microsoft/Phi-3-mini-4k-instruct",
            "tensor_parallel_size": 1
        }
    }
    
    # LLM 설정
    PRELOAD_MODEL: bool = True
    CONTEXT_SIZE: int = 4096
    GPU_LAYERS: int = -1  # -1은 가능한 모든 레이어를 GPU로 로드
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    TOP_K: int = 40
    
    # 로깅 설정
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # 성능 설정
    REQUEST_TIMEOUT: int = 300  # 초 단위
    MAX_CONCURRENT_REQUESTS: int = 5
    
    # 캐싱 설정
    ENABLE_CACHE: bool = True
    CACHE_TTL: int = 3600  # 초 단위 (1시간)
    
    # 보안 설정
    API_KEY_REQUIRED: bool = False
    API_KEY: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# 설정 인스턴스 생성
settings = Settings()

# 모델 디렉터리가 없으면 생성
Path(settings.MODEL_DIR).mkdir(parents=True, exist_ok=True)