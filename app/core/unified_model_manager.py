import logging
import time
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Literal, Union
from pathlib import Path

from app.config import settings
from app.utils.exceptions import ModelNotLoadedError, InferenceError
from app.core.model_config import ModelConfig

# 조건부 임포트
try:
    from app.core.llm_manager import LLMManager
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    
try:
    from app.core.transformers_manager import TransformersManager
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from app.core.vllm_manager import VLLMManager
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)

class UnifiedModelManager:
    """통합 모델 관리 클래스"""
    
    def __init__(self):
        self.llm_manager = None
        self.transformers_manager = None
        self.vllm_manager = None
        self.active_backend: Optional[Literal["llama-cpp", "transformers", "vllm"]] = None
        self.active_model: Optional[str] = None
        self.lock = asyncio.Lock()
        
        # 선택된 백엔드 또는 자동 감지
        self.backend = settings.MODEL_BACKEND
        
        # LLM 매니저 초기화
        if self.backend in ["llama-cpp", "auto"] and LLAMA_CPP_AVAILABLE:
            try:
                self.llm_manager = LLMManager()
                logger.info("llama-cpp-python 백엔드 초기화됨")
                if self.backend == "llama-cpp":
                    self.active_backend = "llama-cpp"
            except Exception as e:
                logger.error(f"llama-cpp-python 백엔드 초기화 실패: {str(e)}")
                if self.backend == "llama-cpp":
                    raise
        
        # Transformers 매니저 초기화
        if self.backend in ["transformers", "auto"] and TRANSFORMERS_AVAILABLE:
            try:
                self.transformers_manager = TransformersManager()
                logger.info("Transformers 백엔드 초기화됨")
                if self.backend == "transformers" or (self.backend == "auto" and not LLAMA_CPP_AVAILABLE and not self.active_backend):
                    self.active_backend = "transformers"
            except Exception as e:
                logger.error(f"Transformers 백엔드 초기화 실패: {str(e)}")
                if self.backend == "transformers":
                    raise
        
        # vLLM 매니저 초기화 - 우선순위 높게 설정
        if self.backend in ["vllm", "auto"] and VLLM_AVAILABLE:
            try:
                self.vllm_manager = VLLMManager()
                logger.info("vLLM 백엔드 초기화됨")
                if self.backend == "vllm" or (self.backend == "auto"):
                    # 자동 모드에서는 vLLM에 높은 우선순위 부여
                    self.active_backend = "vllm"
            except Exception as e:
                logger.error(f"vLLM 백엔드 초기화 실패: {str(e)}")
                if self.backend == "vllm":
                    raise
        
        if not self.active_backend and self.backend == "auto":
            if self.llm_manager:
                self.active_backend = "llama-cpp"
            elif self.transformers_manager:
                self.active_backend = "transformers"
            elif self.vllm_manager:
                self.active_backend = "vllm"
        
        if not self.active_backend:
            raise ImportError("사용 가능한 모델 백엔드가 없습니다. llama-cpp-python, transformers 또는 vllm을 설치하세요.")
        
        logger.info(f"활성 백엔드: {self.active_backend}")
    
    async def load_model(self, model_name: str, config: Optional[ModelConfig] = None, 
                         backend: Optional[Literal["llama-cpp", "transformers", "vllm"]] = None) -> None:
        """모델을 로드합니다."""
        async with self.lock:
            # 백엔드 결정
            if backend:
                use_backend = backend
            else:
                # 모델명으로 백엔드 추론
                if model_name.endswith(".gguf") or model_name in settings.AVAILABLE_MODELS:
                    use_backend = "llama-cpp"
                elif "/" in model_name:
                    # Hugging Face 모델: vLLM 우선, 그 다음 Transformers
                    if VLLM_AVAILABLE and self.vllm_manager:
                        use_backend = "vllm"
                    elif TRANSFORMERS_AVAILABLE and self.transformers_manager:
                        use_backend = "transformers"
                    else:
                        use_backend = self.active_backend
                elif model_name in settings.TRANSFORMERS_MODELS:
                    use_backend = "transformers"
                elif model_name in settings.VLLM_MODELS:
                    use_backend = "vllm"
                else:
                    use_backend = self.active_backend
            
            # vLLM 백엔드를 위한 모델 ID 처리
            if use_backend == "vllm" and model_name in settings.VLLM_MODELS:
                model_info = settings.VLLM_MODELS[model_name]
                actual_model_name = model_info["model_id"]
                
                # 사용자 정의 구성 없는 경우 기본 구성 생성
                if not config:
                    config = ModelConfig()
                
                # 모델 정보에서 추가 구성 적용
                if not config.custom_parameters:
                    config.custom_parameters = {}
                
                for key, value in model_info.items():
                    if key != "model_id":
                        config.custom_parameters[key] = value
            
            # Transformers 백엔드를 위한 모델 ID 처리
            elif use_backend == "transformers" and model_name in settings.TRANSFORMERS_MODELS:
                model_info = settings.TRANSFORMERS_MODELS[model_name]
                actual_model_name = model_info["model_id"]
                
                # 사용자 정의 구성 없는 경우 기본 구성 생성
                if not config:
                    config = ModelConfig()
                
                # 모델 정보에서 추가 구성 적용
                if not config.custom_parameters:
                    config.custom_parameters = {}
                
                for key, value in model_info.items():
                    if key != "model_id":
                        config.custom_parameters[key] = value
            else:
                actual_model_name = model_name
            
            logger.info(f"모델 '{actual_model_name}' 로드 시작 (백엔드: {use_backend})")
            
            if use_backend == "llama-cpp":
                if not self.llm_manager:
                    raise RuntimeError("llama-cpp-python 백엔드가 초기화되지 않았습니다.")
                
                await self.llm_manager.load_model(actual_model_name, config)
                self.active_backend = "llama-cpp"
                self.active_model = actual_model_name
                
            elif use_backend == "transformers":
                if not self.transformers_manager:
                    raise RuntimeError("Transformers 백엔드가 초기화되지 않았습니다.")
                
                await self.transformers_manager.load_model(actual_model_name, config)
                self.active_backend = "transformers"
                self.active_model = actual_model_name
                
            elif use_backend == "vllm":
                if not self.vllm_manager:
                    raise RuntimeError("vLLM 백엔드가 초기화되지 않았습니다.")
                
                await self.vllm_manager.load_model(actual_model_name, config)
                self.active_backend = "vllm"
                self.active_model = actual_model_name
            
            else:
                raise ValueError(f"알 수 없는 백엔드: {use_backend}")
            
            logger.info(f"모델 '{actual_model_name}' 로드 완료 (백엔드: {use_backend})")
    
    async def unload_model(self, model_name: str, backend: Optional[Literal["llama-cpp", "transformers", "vllm"]] = None) -> None:
        """모델을 언로드합니다."""
        async with self.lock:
            # 백엔드 결정
            if backend:
                use_backend = backend
            else:
                # 모델명으로 백엔드 추론
                if model_name.endswith(".gguf") or model_name in settings.AVAILABLE_MODELS:
                    use_backend = "llama-cpp"
                elif "/" in model_name:
                    # 백엔드 확인
                    if self.vllm_manager and model_name in self.vllm_manager.models:
                        use_backend = "vllm"
                    elif self.transformers_manager and model_name in self.transformers_manager.models:
                        use_backend = "transformers"
                    elif VLLM_AVAILABLE and self.vllm_manager:
                        use_backend = "vllm"
                    elif TRANSFORMERS_AVAILABLE and self.transformers_manager:
                        use_backend = "transformers"
                    else:
                        use_backend = self.active_backend
                elif model_name in settings.TRANSFORMERS_MODELS:
                    use_backend = "transformers"
                elif model_name in settings.VLLM_MODELS:
                    use_backend = "vllm"
                else:
                    # 모델이 로드되어 있는지 확인하여 백엔드 결정
                    if self.llm_manager and model_name in self.llm_manager.models:
                        use_backend = "llama-cpp"
                    elif self.transformers_manager and model_name in self.transformers_manager.models:
                        use_backend = "transformers"
                    elif self.vllm_manager and model_name in self.vllm_manager.models:
                        use_backend = "vllm"
                    else:
                        raise ValueError(f"모델 '{model_name}'이(가) 로드되지 않았습니다.")
            
            logger.info(f"모델 '{model_name}' 언로드 시작 (백엔드: {use_backend})")
            
            if use_backend == "llama-cpp":
                if not self.llm_manager:
                    raise RuntimeError("llama-cpp-python 백엔드가 초기화되지 않았습니다.")
                
                await self.llm_manager.unload_model(model_name)
                if self.active_model == model_name and self.active_backend == "llama-cpp":
                    self.active_model = self.llm_manager.active_model
                
            elif use_backend == "transformers":
                if not self.transformers_manager:
                    raise RuntimeError("Transformers 백엔드가 초기화되지 않았습니다.")
                
                await self.transformers_manager.unload_model(model_name)
                if self.active_model == model_name and self.active_backend == "transformers":
                    self.active_model = self.transformers_manager.active_model
                    
            elif use_backend == "vllm":
                if not self.vllm_manager:
                    raise RuntimeError("vLLM 백엔드가 초기화되지 않았습니다.")
                
                await self.vllm_manager.unload_model(model_name)
                if self.active_model == model_name and self.active_backend == "vllm":
                    self.active_model = self.vllm_manager.active_model
            
            else:
                raise ValueError(f"알 수 없는 백엔드: {use_backend}")
            
            logger.info(f"모델 '{model_name}' 언로드 완료 (백엔드: {use_backend})")
    
    async def unload_all_models(self) -> None:
        """모든 모델을 언로드합니다."""
        if self.llm_manager:
            await self.llm_manager.unload_all_models()
        
        if self.transformers_manager:
            await self.transformers_manager.unload_all_models()
            
        if self.vllm_manager:
            await self.vllm_manager.unload_all_models()
        
        self.active_model = None
    
    async def get_embeddings(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """텍스트의 임베딩을 생성합니다."""
        # 모델이 지정되지 않은 경우 현재 활성 모델 사용
        if not model_name:
            if not self.active_model:
                raise ModelNotLoadedError("활성화된 모델이 없습니다.")
            model_name = self.active_model
            backend = self.active_backend
        else:
            # 모델명으로 백엔드 추론
            if model_name.endswith(".gguf") or model_name in settings.AVAILABLE_MODELS:
                backend = "llama-cpp"
            elif "/" in model_name:
                if self.vllm_manager and model_name in self.vllm_manager.models:
                    backend = "vllm"
                elif self.transformers_manager and model_name in self.transformers_manager.models:
                    backend = "transformers"
                elif VLLM_AVAILABLE and self.vllm_manager:
                    backend = "vllm"
                elif TRANSFORMERS_AVAILABLE and self.transformers_manager:
                    backend = "transformers"
                else:
                    backend = self.active_backend
            elif model_name in settings.TRANSFORMERS_MODELS:
                backend = "transformers"
            elif model_name in settings.VLLM_MODELS:
                backend = "vllm"
            else:
                backend = self.active_backend
        
        # 모델 ID 처리
        actual_model_name = model_name
        if backend == "transformers" and model_name in settings.TRANSFORMERS_MODELS:
            actual_model_name = settings.TRANSFORMERS_MODELS[model_name]["model_id"]
        elif backend == "vllm" and model_name in settings.VLLM_MODELS:
            actual_model_name = settings.VLLM_MODELS[model_name]["model_id"]
        
        try:
            if backend == "llama-cpp":
                if not self.llm_manager:
                    raise RuntimeError("llama-cpp-python 백엔드가 초기화되지 않았습니다.")
                
                return await self.llm_manager.get_embeddings(text, actual_model_name)
            
            elif backend == "transformers":
                if not self.transformers_manager:
                    raise RuntimeError("Transformers 백엔드가 초기화되지 않았습니다.")
                
                return await self.transformers_manager.get_embeddings(text, actual_model_name)
                
            elif backend == "vllm":
                if not self.vllm_manager:
                    raise RuntimeError("vLLM 백엔드가 초기화되지 않았습니다.")
                
                return await self.vllm_manager.get_embeddings(text, actual_model_name)
            
            else:
                raise ValueError(f"알 수 없는 백엔드: {backend}")
                
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
        # 모델이 지정되지 않은 경우 현재 활성 모델 사용
        if not model_name:
            if not self.active_model:
                raise ModelNotLoadedError("활성화된 모델이 없습니다.")
            model_name = self.active_model
            backend = self.active_backend
        else:
            # 모델명으로 백엔드 추론
            if model_name.endswith(".gguf") or model_name in settings.AVAILABLE_MODELS:
                backend = "llama-cpp"
            elif "/" in model_name:
                if self.vllm_manager and model_name in self.vllm_manager.models:
                    backend = "vllm"
                elif self.transformers_manager and model_name in self.transformers_manager.models:
                    backend = "transformers"
                elif VLLM_AVAILABLE and self.vllm_manager:
                    backend = "vllm"
                elif TRANSFORMERS_AVAILABLE and self.transformers_manager:
                    backend = "transformers"
                else:
                    backend = self.active_backend
            elif model_name in settings.TRANSFORMERS_MODELS:
                backend = "transformers"
            elif model_name in settings.VLLM_MODELS:
                backend = "vllm"
            else:
                backend = self.active_backend
        
        # 모델 ID 처리
        actual_model_name = model_name
        if backend == "transformers" and model_name in settings.TRANSFORMERS_MODELS:
            actual_model_name = settings.TRANSFORMERS_MODELS[model_name]["model_id"]
        elif backend == "vllm" and model_name in settings.VLLM_MODELS:
            actual_model_name = settings.VLLM_MODELS[model_name]["model_id"]
        
        try:
            if backend == "llama-cpp":
                if not self.llm_manager:
                    raise RuntimeError("llama-cpp-python 백엔드가 초기화되지 않았습니다.")
                
                return await self.llm_manager.generate_completion(
                    prompt=prompt,
                    model_name=actual_model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop
                )
            
            elif backend == "transformers":
                if not self.transformers_manager:
                    raise RuntimeError("Transformers 백엔드가 초기화되지 않았습니다.")
                
                return await self.transformers_manager.generate_completion(
                    prompt=prompt,
                    model_name=actual_model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop
                )
                
            elif backend == "vllm":
                if not self.vllm_manager:
                    raise RuntimeError("vLLM 백엔드가 초기화되지 않았습니다.")
                
                return await self.vllm_manager.generate_completion(
                    prompt=prompt,
                    model_name=actual_model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop
                )
            
            else:
                raise ValueError(f"알 수 없는 백엔드: {backend}")
                
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
        # 모델이 지정되지 않은 경우 현재 활성 모델 사용
        if not model_name:
            if not self.active_model:
                raise ModelNotLoadedError("활성화된 모델이 없습니다.")
            model_name = self.active_model
            backend = self.active_backend
        else:
            # 모델명으로 백엔드 추론
            if model_name.endswith(".gguf") or model_name in settings.AVAILABLE_MODELS:
                backend = "llama-cpp"
            elif "/" in model_name:
                if self.vllm_manager and model_name in self.vllm_manager.models:
                    backend = "vllm"
                elif self.transformers_manager and model_name in self.transformers_manager.models:
                    backend = "transformers"
                elif VLLM_AVAILABLE and self.vllm_manager:
                    backend = "vllm"
                elif TRANSFORMERS_AVAILABLE and self.transformers_manager:
                    backend = "transformers"
                else:
                    backend = self.active_backend
            elif model_name in settings.TRANSFORMERS_MODELS:
                backend = "transformers"
            elif model_name in settings.VLLM_MODELS:
                backend = "vllm"
            else:
                backend = self.active_backend
        
        # 모델 ID 처리
        actual_model_name = model_name
        if backend == "transformers" and model_name in settings.TRANSFORMERS_MODELS:
            actual_model_name = settings.TRANSFORMERS_MODELS[model_name]["model_id"]
        elif backend == "vllm" and model_name in settings.VLLM_MODELS:
            actual_model_name = settings.VLLM_MODELS[model_name]["model_id"]
        
        try:
            if backend == "llama-cpp":
                if not self.llm_manager:
                    raise RuntimeError("llama-cpp-python 백엔드가 초기화되지 않았습니다.")
                
                return await self.llm_manager.generate_chat_completion(
                    messages=messages,
                    model_name=actual_model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop
                )
            
            elif backend == "transformers":
                if not self.transformers_manager:
                    raise RuntimeError("Transformers 백엔드가 초기화되지 않았습니다.")
                
                return await self.transformers_manager.generate_chat_completion(
                    messages=messages,
                    model_name=actual_model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop
                )
                
            elif backend == "vllm":
                if not self.vllm_manager:
                    raise RuntimeError("vLLM 백엔드가 초기화되지 않았습니다.")
                
                return await self.vllm_manager.generate_chat_completion(
                    messages=messages,
                    model_name=actual_model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop
                )
            
            else:
                raise ValueError(f"알 수 없는 백엔드: {backend}")
                
        except Exception as e:
            logger.error(f"채팅 생성 중 오류 발생: {str(e)}")
            raise InferenceError(f"채팅 생성 중 오류: {str(e)}")
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록을 반환합니다."""
        all_models = []
        
        # llama-cpp-python 모델 목록
        if self.llm_manager:
            try:
                llm_models = await self.llm_manager.list_available_models()
                for model in llm_models:
                    model["backend"] = "llama-cpp"
                    model["active_backend"] = (self.active_backend == "llama-cpp")
                all_models.extend(llm_models)
            except Exception as e:
                logger.error(f"llama-cpp-python 모델 목록 조회 중 오류: {str(e)}")
        
        # Transformers 모델 목록
        if self.transformers_manager:
            try:
                transformers_models = await self.transformers_manager.list_available_models()
                for model in transformers_models:
                    model["backend"] = "transformers"
                    model["active_backend"] = (self.active_backend == "transformers")
                all_models.extend(transformers_models)
            except Exception as e:
                logger.error(f"Transformers 모델 목록 조회 중 오류: {str(e)}")
                
        # vLLM 모델 목록
        if self.vllm_manager:
            try:
                vllm_models = await self.vllm_manager.list_available_models()
                for model in vllm_models:
                    model["backend"] = "vllm"
                    model["active_backend"] = (self.active_backend == "vllm")
                all_models.extend(vllm_models)
            except Exception as e:
                logger.error(f"vLLM 모델 목록 조회 중 오류: {str(e)}")
        
        return all_models