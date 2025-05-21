# vLLM이 추가된 로컬 LLM API 서버 아키텍처

이 문서는 vLLM 백엔드가 추가된 로컬 LLM API 서버의 아키텍처를 설명합니다. vLLM은 대규모 언어 모델을 위한 고성능 추론 엔진으로, PagedAttention 메커니즘을 통해 메모리 효율성을 크게 향상시키고 처리량을 높입니다.

## 아키텍처 개요

이 API 서버는 이제 다음 세 가지 백엔드를 통합적으로 지원합니다:

1. **llama-cpp-python**: GGUF 형식의 모델을 위한 백엔드
2. **Hugging Face Transformers**: Hugging Face 모델을 위한 백엔드
3. **vLLM**: 고성능 추론을 위한 최적화된 백엔드

이 세 백엔드는 `UnifiedModelManager`를 통해 통합적으로 관리되며, 각 모델이나 사용 사례에 가장 적합한 백엔드를 자동으로 선택할 수 있습니다.

## vLLM 통합의 주요 장점

vLLM 백엔드를 통합함으로써 다음과 같은 장점을 얻을 수 있습니다:

1. **높은 처리량**: PagedAttention을 통해 최대 24배 더 높은 처리량 제공
2. **메모리 효율성**: 메모리 사용량 최적화로 더 큰 모델 처리 가능
3. **병렬 요청 처리**: 동시에 여러 요청을 효율적으로 처리
4. **텐서 병렬화**: 여러 GPU에 걸쳐 대규모 모델을 분산 실행
5. **양자화 지원**: AWQ, SqueezeLLM, 비트 양자화 등 다양한 양자화 기법 지원

## 통합 아키텍처 상세 설명

### 1. 백엔드 통합 계층

`UnifiedModelManager` 클래스는 세 가지 백엔드를 통합적으로 관리합니다:

```python
class UnifiedModelManager:
    def __init__(self):
        self.llm_manager = None
        self.transformers_manager = None
        self.vllm_manager = None
        self.active_backend = None
        
        # 각 백엔드 초기화
        if LLAMA_CPP_AVAILABLE:
            self.llm_manager = LLMManager()
            
        if TRANSFORMERS_AVAILABLE:
            self.transformers_manager = TransformersManager()
            
        if VLLM_AVAILABLE:
            self.vllm_manager = VLLMManager()
            # 자동 모드에서는 vLLM에 높은 우선순위 부여
```

`UnifiedModelManager`는 사용 가능한 백엔드를 감지하고 설정에 따라 적절한 백엔드를 활성화합니다. 자동 모드에서는 vLLM 백엔드에 가장 높은 우선순위를 부여하여 성능을 최적화합니다.

### 2. vLLM 백엔드

`VLLMManager` 클래스는 vLLM 라이브러리를 사용하여 모델을 로드하고 추론을 수행합니다:

```python
class VLLMManager:
    def __init__(self):
        self.models = {}
        self.sampling_params = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # vLLM 라이브러리 임포트
        from vllm import LLM, SamplingParams
        self.LLM = LLM
        self.SamplingParams = SamplingParams
```

주요 기능:

1. **모델 로드**:
   ```python
   async def load_model(self, model_name, config=None):
       # 텐서 병렬화, 양자화 등 고급 설정
       model = self.LLM(
           model=model_name,
           tensor_parallel_size=tensor_parallel_size,
           quantization=quantization,
           ...
       )
   ```

2. **텍스트 생성**:
   ```python
   async def generate_completion(self, prompt, ...):
       sampling_params = self.SamplingParams(...)
       outputs = await model.generate(prompt, sampling_params)
   ```

3. **채팅 완성**:
   ```python
   async def generate_chat_completion(self, messages, ...):
       # 네이티브 채팅 지원 또는 프롬프트 변환
       if hasattr(model, "chat"):
           outputs = await model.chat(messages, sampling_params)
       else:
           prompt = self._format_chat_messages(messages)
           # 일반 생성 API 사용
   ```

### 3. 모델 자동 라우팅

`UnifiedModelManager`는 모델 이름이나 형식에 따라 자동으로 적절한 백엔드를 선택합니다:

```python
async def load_model(self, model_name, config=None, backend=None):
    # 백엔드 자동 결정
    if not backend:
        if model_name.endswith(".gguf"):
            use_backend = "llama-cpp"
        elif "/" in model_name:  # Hugging Face 모델
            # vLLM 우선, 그 다음 Transformers
            if VLLM_AVAILABLE and self.vllm_manager:
                use_backend = "vllm"
            elif TRANSFORMERS_AVAILABLE:
                use_backend = "transformers"
        # ...
```

이 로직을 통해 사용자는 모델 형식이나 이름만으로도 적절한 백엔드가 자동으로 선택되도록 할 수 있습니다.

### 4. 구성 및 설정

`config.py`에서는 각 백엔드별 설정을 관리합니다:

```python
# vLLM 모델 설정
VLLM_DEFAULT_MODEL: str = "meta-llama/Llama-3-8B-Instruct"
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
    # ...
}
```

이러한 설정을 통해 복잡한 모델 설정을 단순화하고, 사용자 친화적인 별칭(예: "llama3-8b")으로 모델을 참조할 수 있습니다.

## vLLM을 통한 성능 최적화

vLLM은 다음과 같은 방법으로 성능을 최적화합니다:

1. **PagedAttention**: KV 캐시를 페이지 단위로 관리하여 메모리 효율성 향상
2. **연속 배치 처리**: 동시에 여러 요청을 효율적으로 처리
3. **텐서 병렬화**: 여러 GPU에 걸쳐 대규모 모델을 분산 실행
4. **양자화**: 다양한 양자화 기법을 지원하여 메모리 사용량 감소
5. **CUDA 그래프**: 반복적인 연산을 최적화하여 추론 속도 향상

## API 엔드포인트 및 사용 예시

vLLM을 포함한 통합 API 서버는 기존과 동일한 API 엔드포인트를 제공합니다:

1. **채팅 완성**: `/api/v1/chat/completions`
2. **텍스트 완성**: `/api/v1/completions`
3. **임베딩**: `/api/v1/embeddings`
4. **모델 관리**: `/models`, `/models/{model_name}/load`

사용 예시:

```bash
# vLLM 모델을 사용한 채팅 완성
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
      {"role": "user", "content": "인공지능의 미래에 대해 설명해주세요."}
    ],
    "model": "llama3-8b",
    "max_tokens": 500
  }'
```

## 결론

vLLM 통합을 통해 로컬 LLM API 서버는 다음과 같은 이점을 제공합니다:

1. **다양한 백엔드 지원**: GGUF, Transformers, vLLM 모델을 모두 지원
2. **성능 최적화**: vLLM의 PagedAttention을 통한 높은 처리량과 메모리 효율성
3. **자동 백엔드 선택**: 모델 형식이나 이름에 따라 적절한 백엔드 자동 선택
4. **통합 인터페이스**: 일관된 API를 통해 모든 백엔드 사용 가능
5. **확장성**: 다양한 모델 크기와 형식에 대응 가능한 확장 가능한 아키텍처

이 아키텍처를 통해 사용자는 다양한 모델과 백엔드의 장점을 활용하면서도 일관된 인터페이스로 로컬 LLM을 사용할 수 있습니다.# vLLM이 추가된 로컬 LLM API 서버 아키텍처

이 문서는 vLLM 백엔드가 추가된 로컬 LLM API 서버의 아키텍처를 설명합니다. vLLM은 대규모 언어 모델을 위한 고성능 추론 엔진으로, PagedAttention 메커니즘을 통해 메모리 효율성을 크게 향상시키고 처리량을 높입니다.

## 아키텍처 개요

![아키텍처 다이어그램](architecture.png)

이 API 서버는 이제 다음 세 가지 백엔드를 통합적으로 지원합니다:

1. **llama-cpp-python**: GGUF 형식의 모델을 위한 백엔드
2. **Hugging Face Transformers**: Hugging Face 모델을 위한 백엔드
3. **vLLM**: 고성능 추론을 위한 최적화된 백엔드

이 세 백엔드는 `UnifiedModelManager`를 통해 통합적으로 관리되며, 각 모델이나 사용 사례에 가장 적합한 백엔드를 자동으로 선택할 수 있습니다.

## vLLM 통합의 주요 장점

vLLM 백엔드를 통합함으로써 다음과 같은 장점을 얻을 수 있습니다:

1. **높은 처리량**: PagedAttention을 통해 최대 24배 더 높은 처리량 제공
2. **메모리 효율성**: 메모리 사용량 최적화로 더 큰 모델 처리 가능
3. **병렬 요청 처리**: 동시에 여러 요청을 효율적으로 처리
4. **텐서 병렬화**: 여러 GPU에 걸쳐 대규모 모델을 분산 실행
5. **양자화 지원**: AWQ, SqueezeLLM, 비트 양자화 등 다양한 양자화 기법 지원

## vLLM 백엔드 구현

`VLLMManager` 클래스는 vLLM 라이브러리를 활용하여 모델을 관리하고 추론을 수행합니다. 주요 기능은 다음과 같습니다:

1. **모델 로드 및 관리**:
   ```python
   async def load_model(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
       # 모델 로드 로직
       model = self.LLM(
           model=model_name,
           trust_remote_code=True,
           tensor_parallel_size=tensor_parallel_size,  # GPU 병렬화
           quantization=quantization_method,          # 양자화 방식
           # 기타 vLLM 설정
       )
   ```

2. **텍스트 완성 생성**:
   ```python
   async def generate_completion(self, prompt: str, ...) -> Dict[str, Any]:
       # 샘플링 파라미터 설정
       sampling_params = self.SamplingParams(
           temperature=temperature,
           top_p=top_p,
           top_k=top_k,
           max_tokens=max_tokens,
           stop=stop
       )
       
       # vLLM 생성 API 사용
       outputs = await model.generate(prompt, sampling_params)
   ```

3. **채팅 완성 생성**:
   ```python
   async def generate_chat_completion(self, messages: List[Dict[str, str]], ...) -> Dict[str, Any]:
       # vLLM의 네이티브 채팅 지원 활용 또는 프롬프트 변환
       if hasattr(model, "chat"):
           outputs = await model.chat(messages, sampling_params)  
       else:
           # 일반 완성 API를 통한 채팅 에뮬레이션
           prompt = self._format_chat_messages(messages, model_name)
           completion_response = await self.generate_completion(prompt, ...)
   ```

## 통합 모델 매니저 업데이트

`UnifiedModelManager` 클래스는 세 가지 백엔드를 통합적으로 관리하도록 확장되었습니다:

```python
class UnifiedModelManager:
    def __init__(self):
        self.llm_manager = None
        self.transformers_manager = None
        self.vllm_manager = None
        self.active_backend = None
        
        # 각 백엔드 초기화
        if LLAMA_CPP_AVAILABLE:
            self.llm_manager = LLMManager()
            
        if TRANSFORMERS_AVAILABLE:
            self.transformers_manager = TransformersManager()
            
        if VLLM_AVAILABLE:
            self.vllm_manager = VLLMManager()
            # 자동 모드에서는 vLLM에 높은 우선순위 부여