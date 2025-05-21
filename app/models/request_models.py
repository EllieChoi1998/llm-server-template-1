from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union

class ChatMessage(BaseModel):
    """채팅 메시지 모델"""
    role: str = Field(..., description="'system', 'user', 'assistant' 중 하나의 역할")
    content: str = Field(..., description="메시지 내용")
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ['system', 'user', 'assistant']
        if v not in allowed_roles:
            raise ValueError(f"역할은 {', '.join(allowed_roles)} 중 하나여야 합니다")
        return v


class ChatCompletionRequest(BaseModel):
    """채팅 완성 요청 모델"""
    model: Optional[str] = Field(None, description="사용할 모델 이름")
    messages: List[ChatMessage] = Field(..., description="채팅 메시지 목록")
    max_tokens: Optional[int] = Field(None, description="생성할 최대 토큰 수")
    temperature: Optional[float] = Field(None, description="생성 온도 (0.0 ~ 2.0)")
    top_p: Optional[float] = Field(None, description="상위 P 샘플링 (0.0 ~ 1.0)")
    top_k: Optional[int] = Field(None, description="상위 K 샘플링")
    stop: Optional[List[str]] = Field(None, description="생성 중지 토큰 목록")
    stream: Optional[bool] = Field(False, description="스트리밍 응답 여부")
    
    @validator('messages')
    def validate_messages(cls, v):
        if len(v) == 0:
            raise ValueError("메시지 목록은 비어 있을 수 없습니다")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < 0.0 or v > 2.0):
            raise ValueError("온도는 0.0에서 2.0 사이여야 합니다")
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("top_p는 0.0에서 1.0 사이여야 합니다")
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens는 양수여야 합니다")
        return v


class CompletionRequest(BaseModel):
    """텍스트 완성 요청 모델"""
    model: Optional[str] = Field(None, description="사용할 모델 이름")
    prompt: str = Field(..., description="완성 프롬프트")
    max_tokens: Optional[int] = Field(None, description="생성할 최대 토큰 수")
    temperature: Optional[float] = Field(None, description="생성 온도 (0.0 ~ 2.0)")
    top_p: Optional[float] = Field(None, description="상위 P 샘플링 (0.0 ~ 1.0)")
    top_k: Optional[int] = Field(None, description="상위 K 샘플링")
    stop: Optional[List[str]] = Field(None, description="생성 중지 토큰 목록")
    stream: Optional[bool] = Field(False, description="스트리밍 응답 여부")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < 0.0 or v > 2.0):
            raise ValueError("온도는 0.0에서 2.0 사이여야 합니다")
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("top_p는 0.0에서 1.0 사이여야 합니다")
        return v
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens는 양수여야 합니다")
        return v


class EmbeddingRequest(BaseModel):
    """임베딩 요청 모델"""
    model: Optional[str] = Field(None, description="사용할 모델 이름")
    input: Union[str, List[str]] = Field(..., description="임베딩할 텍스트 또는 텍스트 목록")
    
    @validator('input')
    def validate_input(cls, v):
        if isinstance(v, str) and len(v.strip()) == 0:
            raise ValueError("입력 텍스트는 비어 있을 수 없습니다")
        elif isinstance(v, list) and (len(v) == 0 or any(len(t.strip()) == 0 for t in v if isinstance(t, str))):
            raise ValueError("입력 텍스트 목록은 비어 있거나 비어 있는 텍스트를 포함할 수 없습니다")
        return v


class ModelLoadRequest(BaseModel):
    """모델 로드 요청 모델"""
    model_name: str = Field(..., description="로드할 모델 이름")
    config: Optional[Dict[str, Any]] = Field(None, description="모델 구성 (선택 사항)")


class ModelInfoResponse(BaseModel):
    """모델 정보 응답 모델"""
    id: str = Field(..., description="모델 ID")
    filename: str = Field(..., description="모델 파일 이름")
    loaded: bool = Field(..., description="모델 로드 여부")
    active: bool = Field(..., description="현재 활성 모델 여부")
    file_exists: bool = Field(..., description="모델 파일 존재 여부")