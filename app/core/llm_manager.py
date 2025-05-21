import os
import logging
from typing import Dict, List, Optional, Any
import time
import asyncio
from pathlib import Path

from app.config import settings
from app.utils.exceptions import ModelNotLoadedError, InferenceError
from app.core.model_config import ModelConfig

logger = logging.getLogger(__name__)

class LLMManager:
    """LLM 모델 관리 클래스"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        self.model_dir = Path(settings.MODEL_DIR)
        self.lock = asyncio.Lock()
        
        # llama-cpp-python을 여기서 임포트해 의존성 순환 방지
        try:
            from llama_cpp import Llama
            self.Llama = Llama
            logger.info("llama-cpp-python 라이브러리 로드 성공")
        except ImportError:
            logger.error("llama-cpp-python 라이브러리를 설치해야 합니다.")
            raise ImportError("llama-cpp-python 라이브러리가 설치되지 않았습니다.")
    
    async def load_model(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        """모델을 로드합니다."""
        async with self.lock:
            if model_name in self.models:
                logger.info(f"모델 '{model_name}'이(가) 이미 로드되어 있습니다.")
                self.active_model = model_name
                return
            
            start_time = time.time()
            logger.info(f"모델 '{model_name}' 로드 시작...")
            
            try:
                # 모델 파일 경로 찾기
                model_path = None
                if model_name in settings.AVAILABLE_MODELS:
                    model_filename = settings.AVAILABLE_MODELS[model_name]
                    model_path = self.model_dir / model_filename
                else:
                    # 직접 파일 이름을 입력한 경우
                    potential_path = self.model_dir / model_name
                    if potential_path.exists():
                        model_path = potential_path
                
                if model_path is None or not model_path.exists():
                    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_name}")
                
                # 모델 구성 설정
                if config is None:
                    config = ModelConfig()  # 기본 구성 사용
                
                # 모델 로드
                # 비동기 실행을 방해하지 않도록 실행 루프에서 실행
                loop = asyncio.get_event_loop()
                model = await loop.run_in_executor(
                    None,
                    lambda: self.Llama(
                        model_path=str(model_path),
                        n_ctx=config.context_size,
                        n_gpu_layers=config.gpu_layers,
                        n_batch=config.batch_size,
                        seed=config.seed
                    )
                )
                
                # 모델 및 구성 저장
                self.models[model_name] = model
                self.model_configs[model_name] = config
                self.active_model = model_name
                
                end_time = time.time()
                logger.info(f"모델 '{model_name}' 로드 완료! 소요 시간: {end_time - start_time:.2f}초")
                
            except Exception as e:
                logger.error(f"모델 '{model_name}' 로드 중 오류 발생: {str(e)}")
                raise
    
    async def unload_model(self, model_name: str) -> None:
        """모델을 언로드합니다."""
        async with self.lock:
            if model_name not in self.models:
                logger.warning(f"모델 '{model_name}'이(가) 로드되지 않았습니다.")
                return
            
            logger.info(f"모델 '{model_name}' 언로드 중...")
            
            try:
                # 명시적으로 모델 삭제
                del self.models[model_name]
                del self.model_configs[model_name]
                
                if self.active_model == model_name:
                    self.active_model = None
                    # 다른 모델이 있으면 활성화
                    if self.models:
                        self.active_model = next(iter(self.models))
                
                logger.info(f"모델 '{model_name}' 언로드 완료")
                
            except Exception as e:
                logger.error(f"모델 '{model_name}' 언로드 중 오류 발생: {str(e)}")
                raise
    
    async def unload_all_models(self) -> None:
        """모든 모델을 언로드합니다."""
        model_names = list(self.models.keys())
        for model_name in model_names:
            await self.unload_model(model_name)
    
    async def get_model(self, model_name: Optional[str] = None) -> Any:
        """지정된 모델 또는 활성 모델을 반환합니다."""
        if model_name is None:
            model_name = self.active_model
        
        if model_name is None:
            raise ModelNotLoadedError("활성화된 모델이 없습니다.")
        
        if model_name not in self.models:
            # 모델이 로드되지 않았으면 로드 시도
            await self.load_model(model_name)
        
        return self.models[model_name]
    
    async def get_embeddings(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """텍스트의 임베딩을 생성합니다."""
        model = await self.get_model(model_name)
        
        try:
            # 임베딩 생성
            embeddings = model.embed(text)
            return embeddings
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
            raise InferenceError(f"임베딩 생성 중 오류: {str(e)}")
    
    async def generate_completion(
        self, 
        prompt: str, 
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """텍스트 완성을 생성합니다."""
        model = await self.get_model(model_name)
        model_config = self.model_configs.get(model_name or self.active_model, ModelConfig())
        
        # 파라미터 기본값 설정
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        top_p = top_p if top_p is not None else model_config.top_p
        top_k = top_k if top_k is not None else model_config.top_k
        stop = stop or model_config.stop
        
        try:
            # 비동기 실행 루프에서 생성
            loop = asyncio.get_event_loop()
            start_time = time.time()
            
            completion = await loop.run_in_executor(
                None,
                lambda: model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                    echo=False
                )
            )
            
            end_time = time.time()
            
            # 응답 형식화
            result = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name or self.active_model,
                "choices": [
                    {
                        "text": completion["choices"][0]["text"],
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": completion["choices"][0]["finish_reason"]
                    }
                ],
                "usage": {
                    "prompt_tokens": completion["usage"]["prompt_tokens"],
                    "completion_tokens": completion["usage"]["completion_tokens"],
                    "total_tokens": completion["usage"]["total_tokens"]
                },
                "system_info": {
                    "processing_time": end_time - start_time
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류 발생: {str(e)}")
            raise InferenceError(f"텍스트 생성 중 오류: {str(e)}")
    
    async def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """채팅 완성을 생성합니다."""
        model = await self.get_model(model_name)
        model_config = self.model_configs.get(model_name or self.active_model, ModelConfig())
        
        # 파라미터 기본값 설정
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        top_p = top_p if top_p is not None else model_config.top_p
        top_k = top_k if top_k is not None else model_config.top_k
        stop = stop or model_config.stop
        
        try:
            # 채팅 메시지를 프롬프트로 변환
            prompt = self._format_chat_messages(messages)
            
            # 비동기 실행 루프에서 생성
            loop = asyncio.get_event_loop()
            start_time = time.time()
            
            completion = await loop.run_in_executor(
                None,
                lambda: model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                    echo=False
                )
            )
            
            end_time = time.time()
            
            # 응답 형식화
            result = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name or self.active_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": completion["choices"][0]["text"].strip()
                        },
                        "finish_reason": completion["choices"][0]["finish_reason"]
                    }
                ],
                "usage": {
                    "prompt_tokens": completion["usage"]["prompt_tokens"],
                    "completion_tokens": completion["usage"]["completion_tokens"],
                    "total_tokens": completion["usage"]["total_tokens"]
                },
                "system_info": {
                    "processing_time": end_time - start_time
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"채팅 생성 중 오류 발생: {str(e)}")
            raise InferenceError(f"채팅 생성 중 오류: {str(e)}")
    
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """채팅 메시지를 LLM 프롬프트 형식으로 변환합니다."""
        formatted_prompt = ""
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # 시스템 메시지는 특별한 형식으로 처리
                formatted_prompt += f"<s>[SYSTEM] {content} </s>\n"
            elif role == "user":
                formatted_prompt += f"<s>[USER] {content} </s>\n"
            elif role == "assistant":
                formatted_prompt += f"<s>[ASSISTANT] {content} </s>\n"
            else:
                # 알 수 없는 역할은 사용자로 처리
                formatted_prompt += f"<s>[USER] {content} </s>\n"
        
        # 마지막에 어시스턴트 응답을 위한 프롬프트 추가
        formatted_prompt += "<s>[ASSISTANT] "
        
        return formatted_prompt
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록을 반환합니다."""
        # 현재 로드된 모델
        loaded_models = set(self.models.keys())
        
        # 설정에 정의된 모델
        configured_models = set(settings.AVAILABLE_MODELS.keys())
        
        # 모델 디렉터리의 .gguf 파일들
        model_files = set()
        if self.model_dir.exists():
            model_files = {
                f.name for f in self.model_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in ['.gguf', '.bin']
            }
        
        # 모든 모델 정보 합치기
        all_models = []
        
        # 설정에 정의된 모델 추가
        for model_id, filename in settings.AVAILABLE_MODELS.items():
            model_info = {
                "id": model_id,
                "filename": filename,
                "loaded": model_id in loaded_models,
                "active": model_id == self.active_model,
                "file_exists": (self.model_dir / filename).exists()
            }
            all_models.append(model_info)
        
        # 발견된 추가 모델 파일 추가
        for filename in model_files:
            if filename not in [m["filename"] for m in all_models]:
                model_info = {
                    "id": filename,  # ID로 파일명 사용
                    "filename": filename,
                    "loaded": filename in loaded_models,
                    "active": filename == self.active_model,
                    "file_exists": True
                }
                all_models.append(model_info)
        
        return all_models