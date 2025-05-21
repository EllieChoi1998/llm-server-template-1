from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import random

from app.config import settings

class ModelConfig(BaseModel):
    """LLM 모델 구성 클래스"""
    
    # 컨텍스트 크기 (토큰)
    context_size: int = Field(default=settings.CONTEXT_SIZE, ge=512, le=32768)
    
    # 로딩 설정
    gpu_layers: int = Field(default=settings.GPU_LAYERS)  # -1: 모든 레이어
    batch_size: int = Field(default=512, ge=1)  # 배치 크기
    seed: int = Field(default=-1)  # -1: 랜덤 시드
    
    # 생성 설정
    max_tokens: int = Field(default=settings.MAX_TOKENS, ge=1, le=32768)
    temperature: float = Field(default=settings.TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=settings.TOP_P, ge=0.0, le=1.0)
    top_k: int = Field(default=settings.TOP_K, ge=0)
    stop: Optional[List[str]] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    
    # 추가 설정을 위한 커스텀 필드
    custom_parameters: Optional[Dict[str, Any]] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # 시드가 -1이면 랜덤 시드 생성
        if self.seed == -1:
            self.seed = random.randint(0, 2147483647)
        
        # 중지 토큰이 None이면 기본값 설정
        if self.stop is None:
            self.stop = ["</s>", "<s>", "[USER]", "[ASSISTANT]"]
    
    def to_dict(self) -> Dict[str, Any]:
        """구성을 딕셔너리로 변환"""
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """딕셔너리에서 구성 생성"""
        return cls(**config_dict)