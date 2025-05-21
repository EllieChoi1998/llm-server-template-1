import logging
import time
import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import torch
from pathlib import Path

from app.config import settings
from app.utils.exceptions import ModelNotLoadedError, InferenceError
from app.core.model_config import ModelConfig

logger = logging.getLogger(__name__)

class TransformersManager:
    """Hugging Face Transformers 모델 관리 클래스"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        self.lock = asyncio.Lock()
        
        # GPU 사용 가능 여부 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Transformers 모델 장치: {self.device}")
        
        try:
            # transformers 라이브러리 임포트
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
            self.AutoConfig = AutoConfig
            self.AutoModel = AutoModel
            logger.info("Transformers 라이브러리 로드 성공")
        except ImportError:
            logger.error("Transformers 라이브러리를 설치해야 합니다.")
            raise ImportError("Transformers 라이브러리가 설치되지 않았습니다.")
    
    async def load_model(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        """모델을 로드합니다."""
        async with self.lock:
            if model_name in self.models:
                logger.info(f"모델 '{model_name}'이(가) 이미 로드되어 있습니다.")
                self.active_model = model_name
                return
            
            start_time = time.time()
            logger.info(f"Transformers 모델 '{model_name}' 로드 시작...")
            
            try:
                # 모델 구성 설정
                if config is None:
                    config = ModelConfig()  # 기본 구성 사용
                
                # 모델과 토크나이저 로드 (비동기 루프에서 실행)
                loop = asyncio.get_event_loop()
                
                # 토크나이저 먼저 로드
                tokenizer = await loop.run_in_executor(
                    None,
                    lambda: self.AutoTokenizer.from_pretrained(
                        model_name,
                        use_fast=True
                    )
                )
                
                # 8비트 또는 4비트 양자화 옵션
                load_in_8bit = config.custom_parameters.get("load_in_8bit", False) if config.custom_parameters else False
                load_in_4bit = config.custom_parameters.get("load_in_4bit", False) if config.custom_parameters else False
                
                # 모델 로드
                model = await loop.run_in_executor(
                    None,
                    lambda: self.AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map="auto" if self.device == "cuda" else None,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        load_in_8bit=load_in_8bit,
                        load_in_4bit=load_in_4bit,
                        trust_remote_code=True
                    )
                )
                
                # 모델, 토크나이저 및 구성 저장
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
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
                # GPU 메모리 명시적 정리
                if self.device == "cuda":
                    self.models[model_name].to("cpu")
                    torch.cuda.empty_cache()
                
                # 모델 및 토크나이저 삭제
                del self.models[model_name]
                del self.tokenizers[model_name]
                del self.model_configs[model_name]
                
                # 활성 모델 재설정
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
    
    async def get_model_and_tokenizer(self, model_name: Optional[str] = None) -> Tuple[Any, Any]:
        """지정된 모델과 토크나이저를 반환합니다."""
        if model_name is None:
            model_name = self.active_model
        
        if model_name is None:
            raise ModelNotLoadedError("활성화된 모델이 없습니다.")
        
        if model_name not in self.models:
            # 모델이 로드되지 않았으면 로드 시도
            await self.load_model(model_name)
        
        return self.models[model_name], self.tokenizers[model_name]
    
    async def get_embeddings(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """텍스트의 임베딩을 생성합니다."""
        model, tokenizer = await self.get_model_and_tokenizer(model_name)
        
        try:
            # 토큰화
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 임베딩 생성 (평균 풀링)
            with torch.no_grad():
                outputs = model.get_input_embeddings()(inputs.input_ids)
                # 패딩 토큰 마스킹 (패딩 토큰 ID는 일반적으로 0)
                mask = inputs.input_ids != tokenizer.pad_token_id
                mask = mask.unsqueeze(-1).expand(outputs.size()).float()
                # 마스킹된 임베딩 평균
                embeddings = torch.sum(outputs * mask, 1) / torch.sum(mask, 1)
                # numpy 배열로 변환
                embedding_np = embeddings[0].cpu().numpy().tolist()
            
            return embedding_np
            
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
        model, tokenizer = await self.get_model_and_tokenizer(model_name)
        model_config = self.model_configs.get(model_name or self.active_model, ModelConfig())
        
        # 파라미터 기본값 설정
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        top_p = top_p if top_p is not None else model_config.top_p
        top_k = top_k if top_k is not None else model_config.top_k
        stop = stop or model_config.stop
        
        try:
            # 입력 토큰화
            inputs = tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            input_length = inputs.input_ids.shape[1]
            
            # 생성 시작 시간
            start_time = time.time()
            
            # 텍스트 생성
            loop = asyncio.get_event_loop()
            generation_output = await loop.run_in_executor(
                None,
                lambda: model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask if hasattr(inputs, 'attention_mask') else None,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            )
            
            # 생성된 텍스트 디코딩
            generated_tokens = generation_output.sequences[0, input_length:]
            completion_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 생성된 토큰 수
            completion_length = len(generated_tokens)
            
            # 종료 이유 결정
            finish_reason = "stop"
            if completion_length >= max_tokens:
                finish_reason = "length"
            
            # 지정된 stop 토큰에 의한 종료 여부 확인
            if stop:
                for stop_token in stop:
                    if stop_token in completion_text:
                        finish_reason = "stop"
                        # stop 토큰 위치까지만 텍스트 자르기
                        completion_text = completion_text.split(stop_token)[0]
                        break
            
            # 응답 형식화
            end_time = time.time()
            result = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name or self.active_model,
                "choices": [
                    {
                        "text": completion_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": input_length,
                    "completion_tokens": completion_length,
                    "total_tokens": input_length + completion_length
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
        model, tokenizer = await self.get_model_and_tokenizer(model_name)
        model_config = self.model_configs.get(model_name or self.active_model, ModelConfig())
        
        # 파라미터 기본값 설정
        max_tokens = max_tokens or model_config.max_tokens
        temperature = temperature if temperature is not None else model_config.temperature
        top_p = top_p if top_p is not None else model_config.top_p
        top_k = top_k if top_k is not None else model_config.top_k
        stop = stop or model_config.stop
        
        try:
            # 채팅 메시지를 프롬프트로 변환
            prompt = self._format_chat_messages(messages, tokenizer)
            
            # 입력 토큰화
            inputs = tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            input_length = inputs.input_ids.shape[1]
            
            # 생성 시작 시간
            start_time = time.time()
            
            # 텍스트 생성
            loop = asyncio.get_event_loop()
            generation_output = await loop.run_in_executor(
                None,
                lambda: model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    attention_mask=inputs.attention_mask if hasattr(inputs, 'attention_mask') else None,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            )
            
            # 생성된 텍스트 디코딩
            generated_tokens = generation_output.sequences[0, input_length:]
            completion_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 생성된 토큰 수
            completion_length = len(generated_tokens)
            
            # 종료 이유 결정
            finish_reason = "stop"
            if completion_length >= max_tokens:
                finish_reason = "length"
            
            # 지정된 stop 토큰에 의한 종료 여부 확인
            if stop:
                for stop_token in stop:
                    if stop_token in completion_text:
                        finish_reason = "stop"
                        # stop 토큰 위치까지만 텍스트 자르기
                        completion_text = completion_text.split(stop_token)[0]
                        break
            
            # 응답 형식화
            end_time = time.time()
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
                            "content": completion_text.strip()
                        },
                        "finish_reason": finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": input_length,
                    "completion_tokens": completion_length,
                    "total_tokens": input_length + completion_length
                },
                "system_info": {
                    "processing_time": end_time - start_time
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"채팅 생성 중 오류 발생: {str(e)}")
            raise InferenceError(f"채팅 생성 중 오류: {str(e)}")
    
    def _format_chat_messages(self, messages: List[Dict[str, str]], tokenizer) -> str:
        """채팅 메시지를 Transformers 모델에 맞는 프롬프트 형식으로 변환합니다."""
        formatted_prompt = ""
        
        # 대화 형식 가이드: 모델 타입이나 토크나이저 유형에 따라 맞춤 설정
        if tokenizer.name_or_path.startswith("meta-llama/Llama") or tokenizer.name_or_path.startswith("lmsys/vicuna"):
            # Llama 2, Llama 3 스타일 프롬프트
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            
            system_msg = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_msg += msg["content"] + "\n"
            
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
            
        elif tokenizer.name_or_path.startswith("google/gemma") or "gemma" in tokenizer.name_or_path.lower():
            # Gemma 스타일 프롬프트
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
            
        elif tokenizer.name_or_path.startswith("mistralai/Mistral") or "mistral" in tokenizer.name_or_path.lower():
            # Mistral 스타일 프롬프트
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
            
        else:
            # 기본 채팅 프롬프트 형식
            system_content = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_content += msg["content"] + "\n"
            
            if system_content:
                formatted_prompt = f"System: {system_content}\n"
            else:
                formatted_prompt = ""
                
            for msg in messages:
                if msg["role"] == "system":
                    continue  # 이미 처리됨
                    
                if msg["role"] == "user":
                    formatted_prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    formatted_prompt += f"Assistant: {msg['content']}\n"
            
            # 모델 응답 유도
            formatted_prompt += "Assistant: "
        
        return formatted_prompt
    
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록을 반환합니다."""
        # 현재 로드된 모델
        loaded_models = set(self.models.keys())
        
        # Hugging Face 모델 목록 (예시)
        recommended_models = [
            {"id": "meta-llama/Llama-3-8B-Instruct", "type": "transformers"},
            {"id": "google/gemma-7b-it", "type": "transformers"},
            {"id": "mistralai/Mistral-7B-Instruct-v0.2", "type": "transformers"},
            {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "type": "transformers"}
        ]
        
        # 모든 모델 정보 합치기
        all_models = []
        
        # 추천 모델 추가
        for model_info in recommended_models:
            model_id = model_info["id"]
            model_info_expanded = {
                "id": model_id,
                "name": model_id.split("/")[-1],
                "type": "transformers",
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
                    "type": "transformers",
                    "loaded": True,
                    "active": model_id == self.active_model
                }
                all_models.append(model_info)
        
        return all_models