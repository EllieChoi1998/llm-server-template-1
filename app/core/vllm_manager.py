import logging
import time
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path

from app.config import settings
from app.utils.exceptions import ModelNotLoadedError, InferenceError
from app.core.model_config import ModelConfig

logger = logging.getLogger(__name__)

class VLLMManager:
    """vLLM 모델 관리 클래스"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.sampling_params: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        self.lock = asyncio.Lock()
        
        try:
            # vLLM 라이브러리 임포트
            from vllm import LLM, SamplingParams
            self.LLM = LLM
            self.SamplingParams = SamplingParams
            logger.info("vLLM 라이브러리 로드 성공")
            
            # AsyncLLM 임포트 시도 (일부 vLLM 버전에서는 지원하지 않을 수 있음)
            try:
                from vllm import AsyncLLM
                self.AsyncLLM = AsyncLLM
                self.supports_async = True
                logger.info("vLLM AsyncLLM 지원 확인됨")
            except ImportError:
                self.AsyncLLM = None
                self.supports_async = False
                logger.info("vLLM AsyncLLM 지원되지 않음, 동기 인터페이스 사용")
                
        except ImportError:
            logger.error("vLLM 라이브러리를 설치해야 합니다.")
            raise ImportError("vLLM 라이브러리가 설치되지 않았습니다.")
    
    async def load_model(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        """모델을 로드합니다."""
        async with self.lock:
            if model_name in self.models:
                logger.info(f"모델 '{model_name}'이(가) 이미 로드되어 있습니다.")
                self.active_model = model_name
                return
            
            start_time = time.time()
            logger.info(f"vLLM 모델 '{model_name}' 로드 시작...")
            
            try:
                # 모델 구성 설정
                if config is None:
                    config = ModelConfig()  # 기본 구성 사용
                
                # GPU 설정
                gpu_config = {}
                if hasattr(config, "tensor_parallel_size") and config.tensor_parallel_size > 1:
                    gpu_config["tensor_parallel_size"] = config.tensor_parallel_size
                
                # 양자화 설정
                if config.custom_parameters and "quantization" in config.custom_parameters:
                    quant_type = config.custom_parameters["quantization"]
                    if quant_type == "awq":
                        gpu_config["quantization"] = "awq"
                    elif quant_type == "sq":
                        gpu_config["quantization"] = "squeezellm"
                    elif quant_type in ["int8", "8bit"]:
                        gpu_config["quantization"] = "int8"
                    elif quant_type in ["int4", "4bit"]:
                        gpu_config["quantization"] = "int4"
                
                # 스왑 공간 설정
                if config.custom_parameters and "swap_space" in config.custom_parameters:
                    gpu_config["swap_space"] = config.custom_parameters["swap_space"]
                
                # 캐시 설정
                if config.custom_parameters and "max_model_len" in config.custom_parameters:
                    gpu_config["max_model_len"] = config.custom_parameters["max_model_len"]
                
                # 모델 로드 (비동기 루프에서 실행)
                if self.supports_async:
                    loop = asyncio.get_event_loop()
                    model = await loop.run_in_executor(
                        None,
                        lambda: self.AsyncLLM(
                            model=model_name,
                            trust_remote_code=True,
                            **gpu_config
                        )
                    )
                else:
                    loop = asyncio.get_event_loop()
                    model = await loop.run_in_executor(
                        None,
                        lambda: self.LLM(
                            model=model_name,
                            trust_remote_code=True,
                            **gpu_config
                        )
                    )
                
                # 샘플링 파라미터 설정
                sampling_params = self.SamplingParams(
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    max_tokens=config.max_tokens,
                    presence_penalty=config.presence_penalty,
                    frequency_penalty=config.frequency_penalty
                )
                
                # 모델 및 구성 저장
                self.models[model_name] = model
                self.sampling_params[model_name] = sampling_params
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
                # vLLM 모델의 명시적 정리
                del self.models[model_name]
                del self.sampling_params[model_name]
                del self.model_configs[model_name]
                
                # 활성 모델 재설정
                if self.active_model == model_name:
                    self.active_model = None
                    # 다른 모델이 있으면 활성화
                    if self.models:
                        self.active_model = next(iter(self.models))
                
                # GPU 메모리 정리 시도
                try:
                    import torch
                    torch.cuda.empty_cache()
                    logger.info("CUDA 캐시 정리됨")
                except:
                    pass
                
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
            # vLLM의 일부 버전은 임베딩을 직접 지원하지 않을 수 있음
            if hasattr(model, "get_embeddings"):
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, 
                    lambda: model.get_embeddings([text])[0]
                )
                return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
            else:
                raise NotImplementedError("이 vLLM 모델은 임베딩 생성을 지원하지 않습니다.")
            
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
        
        # 기본 샘플링 파라미터 가져오기
        base_sampling_params = self.sampling_params.get(model_name or self.active_model)
        
        # 파라미터 기본값 설정
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        top_p = top_p if top_p is not None else model_config.top_p
        top_k = top_k if top_k is not None else model_config.top_k
        stop = stop or model_config.stop
        
        try:
            # 샘플링 파라미터 생성
            sampling_params = self.SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop=stop
            )
            
            # 생성 시작 시간
            start_time = time.time()
            
            # 텍스트 생성
            loop = asyncio.get_event_loop()
            
            if self.supports_async and hasattr(model, "generate"):
                # 비동기 생성 사용
                outputs = await model.generate(prompt, sampling_params)
                output = outputs[0]  # 첫 번째 결과만 사용
            else:
                # 동기 생성 사용
                outputs = await loop.run_in_executor(
                    None,
                    lambda: model.generate([prompt], sampling_params)
                )
                output = outputs[0]  # 첫 번째 결과만 사용
            
            # 생성된 텍스트 추출
            if hasattr(output, "outputs"):
                # 새로운 vLLM 버전
                generated_text = output.outputs[0].text
                prompt_tokens = output.prompt_token_ids.shape[0] if hasattr(output, "prompt_token_ids") else len(output.prompt_token_ids)
                completion_tokens = len(output.outputs[0].token_ids)
                finish_reason = output.outputs[0].finish_reason
            else:
                # 구 버전 vLLM
                generated_text = output.text
                prompt_tokens = len(output.prompt_token_ids)
                completion_tokens = len(output.generated_token_ids)
                finish_reason = "stop" if output.finished else "length"
            
            end_time = time.time()
            
            # 응답 형식화
            result = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name or self.active_model,
                "choices": [
                    {
                        "text": generated_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
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
            # 채팅 메시지를 적절한 형식으로 변환
            if hasattr(model, "chat_completion") or (self.supports_async and hasattr(model, "chat")):
                # 네이티브 채팅 완성 지원 (OpenAI 형식)
                formatted_messages = messages
            else:
                # 일반 텍스트 생성 사용
                prompt = self._format_chat_messages(messages, model_name)
                
                # 일반 완성 API 사용
                completion_response = await self.generate_completion(
                    prompt=prompt,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop
                )
                
                # 채팅 형식으로 변환
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
                                "content": completion_response["choices"][0]["text"].strip()
                            },
                            "finish_reason": completion_response["choices"][0]["finish_reason"]
                        }
                    ],
                    "usage": completion_response["usage"],
                    "system_info": completion_response["system_info"]
                }
                
                return result
            
            # 네이티브 채팅 완성 사용
            # 샘플링 파라미터 생성
            sampling_params = self.SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                stop=stop
            )
            
            # 생성 시작 시간
            start_time = time.time()
            
            # 채팅 완성 생성
            loop = asyncio.get_event_loop()
            
            if self.supports_async and hasattr(model, "chat"):
                # 비동기 채팅 생성 사용
                outputs = await model.chat(messages, sampling_params)
                output = outputs[0]  # 첫 번째 결과만 사용
            else:
                # 동기 채팅 생성 사용
                outputs = await loop.run_in_executor(
                    None,
                    lambda: model.chat_completion(messages, sampling_params)
                )
                output = outputs
            
            # 생성된 텍스트 추출
            if hasattr(output, "choices"):
                # OpenAI 형식의 응답
                generated_text = output.choices[0].message.content
                finish_reason = output.choices[0].finish_reason
                usage = output.usage.dict() if hasattr(output.usage, "dict") else output.usage
            else:
                # 커스텀 형식의 응답
                generated_text = output.output_text
                finish_reason = "stop" if output.finished else "length"
                usage = {
                    "prompt_tokens": output.prompt_token_count,
                    "completion_tokens": output.completion_token_count,
                    "total_tokens": output.prompt_token_count + output.completion_token_count
                }
            
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
                            "content": generated_text
                        },
                        "finish_reason": finish_reason
                    }
                ],
                "usage": usage,
                "system_info": {
                    "processing_time": end_time - start_time
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"채팅 생성 중 오류 발생: {str(e)}")
            raise InferenceError(f"채팅 생성 중 오류: {str(e)}")
    
    def _format_chat_messages(self, messages: List[Dict[str, str]], model_name: Optional[str] = None) -> str:
        """채팅 메시지를 vLLM 모델에 맞는 프롬프트 형식으로 변환합니다."""
        if model_name is None:
            model_name = self.active_model
        
        # 모델 이름에 따라 적절한 프롬프트 포맷 선택
        if "llama" in model_name.lower():
            # Llama 스타일 프롬프트
            return self._format_messages_llama(messages)
        elif "mistral" in model_name.lower():
            # Mistral 스타일 프롬프트
            return self._format_messages_mistral(messages)
        elif "gemma" in model_name.lower():
            # Gemma 스타일 프롬프트
            return self._format_messages_gemma(messages)
        else:
            # 기본 채팅 형식
            return self._format_messages_default(messages)
    
    def _format_messages_llama(self, messages: List[Dict[str, str]]) -> str:
        """Llama 모델용 채팅 메시지 형식"""
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        
        system_msg = ""
        for msg in messages:
            if msg["role"] == "system":
                system_msg += msg["content"] + "\n"
        
        formatted_prompt = ""
        
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                continue  # 시스템 메시지는 별도로 처리함
                
            if msg["role"] == "user":
                if i == 0 or messages[i-1]["role"] == "assistant":
                    # 첫 번째 사용자 메시지이거나, 이전 메시지가 어시스턴트인 경우 새로운 [INST] 블록 시작
                    if system_msg and i == 0:
                        # 시스템 메시지가 있고 첫 번째 사용자 메시지인 경우
                        formatted_prompt += f"{B_INST} {B_SYS}{system_msg}{E_SYS}{msg['content']} {E_INST} "
                    else:
                        # 시스템 메시지가 없거나 첫 번째 사용자 메시지가 아닌 경우
                        formatted_prompt += f"{B_INST} {msg['content']} {E_INST} "
                else:
                    # 이전 메시지도 사용자인 경우 콘텐츠만 추가
                    formatted_prompt += f"{msg['content']} "
            
            elif msg["role"] == "assistant":
                formatted_prompt += f"{msg['content']} "
        
        return formatted_prompt
    
    def _format_messages_mistral(self, messages: List[Dict[str, str]]) -> str:
        """Mistral 모델용 채팅 메시지 형식"""
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
        
        if system_content:
            formatted_prompt = f"<s>[INST] {system_content} [/INST]"
        else:
            formatted_prompt = ""
        
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                continue  # 이미 처리됨
                
            if msg["role"] == "user":
                if i == 0 and not system_content:
                    formatted_prompt += f"<s>[INST] {msg['content']} [/INST]"
                else:
                    formatted_prompt += f"[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted_prompt += f" {msg['content']} </s>"
        
        # 마지막이 사용자 메시지인 경우, 모델 응답 위한 태그 추가
        if messages[-1]["role"] == "user":
            formatted_prompt += " "
        
        return formatted_prompt
    
    def _format_messages_gemma(self, messages: List[Dict[str, str]]) -> str:
        """Gemma 모델용 채팅 메시지 형식"""
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
        
        if system_content:
            formatted_prompt = f"<start_of_turn>system\n{system_content}<end_of_turn>\n"
        else:
            formatted_prompt = ""
        
        for msg in messages:
            if msg["role"] == "system":
                continue  # 이미 처리됨
                
            if msg["role"] == "user":
                formatted_prompt += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
        
        # 모델 응답 유도
        formatted_prompt += "<start_of_turn>model\n"
        
        return formatted_prompt
    
    def _format_messages_default(self, messages: List[Dict[str, str]]) -> str:
        """기본 채팅 메시지 형식"""
        formatted_prompt = ""
        
        for msg in messages:
            if msg["role"] == "system":
                formatted_prompt += f"System: {msg['content']}\n"
            elif msg["role"] == "user":
                formatted_prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"Assistant: {msg['content']}\n"
        
        # 마지막 메시지가 사용자인 경우 어시스턴트 프롬프트 추가
        if messages[-1]["role"] == "user":
            formatted_prompt += "Assistant: "
        
        return formatted_prompt
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록을 반환합니다."""
        # 현재 로드된 모델
        loaded_models = set(self.models.keys())
        
        # Hugging Face 추천 모델 (vLLM에서 테스트된 모델들)
        recommended_models = [
            {"id": "meta-llama/Llama-3-8B-Instruct", "type": "vllm"},
            {"id": "meta-llama/Llama-3-70B-Instruct", "type": "vllm"},
            {"id": "meta-llama/Llama-2-7b-chat-hf", "type": "vllm"},
            {"id": "mistralai/Mistral-7B-Instruct-v0.2", "type": "vllm"},
            {"id": "microsoft/Phi-3-mini-4k-instruct", "type": "vllm"},
            {"id": "google/gemma-7b-it", "type": "vllm"}
        ]
        
        # 모든 모델 정보 합치기
        all_models = []
        
        # 추천 모델 추가
        for model_info in recommended_models:
            model_id = model_info["id"]
            model_info_expanded = {
                "id": model_id,
                "name": model_id.split("/")[-1],
                "type": "vllm",
                "loaded": model_id in loaded_models,
                "active": model_id == self.active_model
            }
            all_models.append(model_info_expanded)
        
        # 로드된 모델 중 추천 목록에 없는 모델 추가
        for model_id in loaded_models:
            if not any(model["id"] == model_id for model in all_models):
                model_info = {
                    "id": model_id,
                    "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                    "type": "vllm",
                    "loaded": True,
                    "active": model_id == self.active_model
                }
                all_models.append(model_info)
        
        return all_models