# 로컬 LLM API 서버

FastAPI를 사용하여 로컬 LLM(Large Language Model)을 API 서버로 제공하는 프로젝트입니다. GGUF 형식의 모델과 Hugging Face Transformers 모델을 모두 지원합니다.

## 주요 기능

- **로컬 LLM 모델 실행**: 다양한 모델 백엔드 지원
  - **GGUF 모델**: llama-cpp-python을 통한 GGUF 형식 모델 실행
  - **Transformers 모델**: Hugging Face Transformers를 통한 모델 실행
  - **통합 관리**: 하나의 API로 두 백엔드 모두 사용 가능
- **OpenAI 호환 API**: OpenAI 형식의 API 엔드포인트 제공
- **텍스트 완성(Completion)**: 텍스트 완성 기능
- **채팅(Chat)**: 채팅 기반의 대화 기능
- **임베딩(Embedding)**: 텍스트 임베딩 생성 기능
- **GPU 가속**: CUDA를 통한 GPU 가속 지원
- **Docker 지원**: 간편한 배포를 위한 Docker 지원

## 시스템 요구사항

- Python 3.9+
- CUDA 지원 GPU (NVIDIA) (선택적)
- Docker 및 Docker Compose (선택적)

## 설치 및 실행

### 방법 1: Docker Compose 사용

1. 저장소를 클론합니다:
   ```bash
   git clone https://github.com/yourusername/local-llm-api.git
   cd local-llm-api
   ```

2. 모델을 다운로드하고 `models` 디렉토리에 저장합니다:
   ```bash
   mkdir -p models
   # 여기에 원하는 GGUF 모델을 다운로드
   ```

3. Docker Compose로 실행합니다:
   ```bash
   docker-compose up -d
   ```

4. API 서버는 `http://localhost:8000`에서 접근할 수 있습니다.

### 방법 2: 직접 실행

1. 저장소를 클론합니다:
   ```bash
   git clone# 로컬 LLM API 서버

FastAPI를 사용하여 로컬 LLM(Large Language Model)을 API 서버로 제공하는 프로젝트입니다.

## 주요 기능

- **로컬 LLM 모델 실행**: GGUF 포맷의 모델을 로컬에서 실행
- **OpenAI 호환 API**: OpenAI 형식의 API 엔드포인트 제공
- **텍스트 완성(Completion)**: 텍스트 완성 기능
- **채팅(Chat)**: 채팅 기반의 대화 기능
- **임베딩(Embedding)**: 텍스트 임베딩 생성 기능
- **GPU 가속**: CUDA를 통한 GPU 가속 지원
- **Docker 지원**: 간편한 배포를 위한 Docker 지원

## 시스템 요구사항

- Python 3.9+
- CUDA 지원 GPU (NVIDIA) (선택적)
- Docker 및 Docker Compose (선택적)

## 설치 및 실행

### 방법 1: Docker Compose 사용

1. 저장소를 클론합니다:
   ```bash
   git clone https://github.com/yourusername/local-llm-api.git
   cd local-llm-api
   ```

2. 모델을 다운로드하고 `models` 디렉토리에 저장합니다:
   ```bash
   mkdir -p models
   # 여기에 원하는 GGUF 모델을 다운로드
   ```

3. Docker Compose로 실행합니다:
   ```bash
   docker-compose up -d
   ```

4. API 서버는 `http://localhost:8000`에서 접근할 수 있습니다.

### 방법 2: 직접 실행

1. 저장소를 클론합니다:
   ```bash
   git clone https://github.com/yourusername/local-llm-api.git
   cd local-llm-api
   ```

2. 가상 환경을 생성하고 활성화합니다:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

3. 필요한 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```

4. 모델을 다운로드하고 `models` 디렉토리에 저장합니다.

5. 환경 변수를 설정합니다 (또는 `.env` 파일을 편집합니다).

6. 서버를 실행합니다:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API 사용 방법

API 문서는 `http://localhost:8000/docs`에서 확인할 수 있습니다.

### 텍스트 완성 (Completion)

```bash
curl -X POST "http://localhost:8000/api/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "안녕하세요, 저는",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### 채팅 완성 (Chat Completion)

```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
      {"role": "user", "content": "안녕하세요, 저는 김철수입니다."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

### 임베딩 생성 (Embedding)

```bash
curl -X POST "http://localhost:8000/api/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "안녕하세요, 저는 김철수입니다."
  }'
```

## 모델 관리

### 사용 가능한 모델 목록 확인

```bash
curl -X GET "http://localhost:8000/models"
```

### 새 모델 로드

```bash
curl -X POST "http://localhost:8000/models/llama3-8b/load"
```

## 환경 변수 설정

주요 환경 변수:

- `MODEL_DIR`: 모델 파일이 저장된 디렉토리 경로
- `DEFAULT_MODEL`: 기본적으로 로드할 모델 이름
- `GPU_LAYERS`: GPU에 로드할 레이어 수 (-1은 모든 레이어)
- `PRELOAD_MODEL`: 시작 시 모델 자동 로드 여부
- `API_KEY_REQUIRED`: API 키 인증 필요 여부
- `API_KEY`: API 키

모든 설정은 `.env` 파일에서 구성할 수 있습니다.

-----------

# 로컬 LLM API 서버 아키텍처

이 문서는 로컬 LLM API 서버의 아키텍처를 설명합니다. 이 서버는 GGUF 형식의 모델과 Hugging Face Transformers 모델을 모두 지원하며, FastAPI를 사용하여 RESTful API를 제공합니다.

## 아키텍처 개요

로컬 LLM API 서버는 다음과 같은 주요 구성 요소로 이루어져 있습니다:

1. **API 서버 (FastAPI)**: 클라이언트 요청을 처리하는 웹 서버
2. **API 라우트**: 엔드포인트 정의 및 요청 처리
3. **서비스 계층**: 비즈니스 로직 처리
4. **모델 관리 계층**: 다양한 모델 백엔드 관리
5. **핵심 구성 요소**: 설정, 로깅, 예외 처리 등

## 컴포넌트 상세 설명

### 1. API 서버 (FastAPI)

FastAPI 프레임워크를 사용하여 고성능 비동기 웹 서버를 구현합니다. CORS, 미들웨어, 의존성 주입, OpenAPI 문서 등 FastAPI의 다양한 기능을 활용합니다.

```python
# app/main.py
from fastapi import FastAPI
from app.core.unified_model_manager import UnifiedModelManager

app = FastAPI(
    title="Local LLM API",
    description="로컬 LLM을 위한 FastAPI 서버",
    version="0.1.0",
)

@app.on_event("startup")
async def startup_event():
    # 통합 모델 매니저 초기화
    model_manager = UnifiedModelManager()
    app.state.model_manager = model_manager
    
    # 모델 사전 로드
    if settings.PRELOAD_MODEL:
        await model_manager.load_model(default_model)
```

### 2. API 라우트

#### 2.1 건강 체크 및 모델 관리 API

서버 상태 및 모델 관리를 위한 API 엔드포인트:

- `GET /health`: 서버 상태 확인
- `GET /models`: 사용 가능한 모델 목록
- `POST /models/{model_name}/load`: 특정 모델 로드

#### 2.2 채팅 API

채팅 완성을 위한 API 엔드포인트:

- `POST /api/v1/chat/completions`: 채팅 메시지에 대한 응답 생성

```python
@router.post("/chat/completions")
async def create_chat_completion(
    request: Request,
    chat_request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    # 모델 매니저 가져오기
    model_manager = request.app.state.model_manager
    
    # 채팅 서비스 생성
    chat_service = ChatService(model_manager)
    
    # 채팅 완성 생성
    response = await chat_service.generate_chat_completion(
        messages=chat_request.messages,
        model=chat_request.model,
        max_tokens=chat_request.max_tokens,
        temperature=chat_request.temperature,
        top_p=chat_request.top_p,
        top_k=chat_request.top_k,
        stop=chat_request.stop
    )
    
    return response
```

#### 2.3 완성 API

텍스트 완성을 위한 API 엔드포인트:

- `POST /api/v1/completions`: 프롬프트에 대한 텍스트 완성 생성

#### 2.4 임베딩 API

텍스트 임베딩을 위한 API 엔드포인트:

- `POST /api/v1/embeddings`: 텍스트에 대한 임베딩 벡터 생성

### 3. 서비스 계층

#### 3.1 채팅 서비스

채팅 완성을 위한 비즈니스 로직:

```python
class ChatService:
    """채팅 완성 서비스"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    async def generate_chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """채팅 완성을 생성합니다."""
        # 요청 검증
        if not messages:
            raise InvalidRequestError("메시지 목록은 비어 있을 수 없습니다")
        
        # 메시지 형식 변환
        formatted_messages = [msg.dict() for msg in messages]
        
        # 모델 매니저를 통해 채팅 완성 생성
        response = await self.model_manager.generate_chat_completion(
            messages=formatted_messages,
            model_name=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop
        )
        
        return response
```

#### 3.2 완성 서비스

텍스트 완성을 위한 비즈니스 로직:

```python
class CompletionService:
    """텍스트 완성 서비스"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """텍스트 완성을 생성합니다."""
        # 요청 검증
        if not prompt or not prompt.strip():
            raise InvalidRequestError("프롬프트는 비어 있을 수 없습니다")
        
        # 모델 매니저를 통해 텍스트 완성 생성
        response = await self.model_manager.generate_completion(
            prompt=prompt,
            model_name=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop
        )
        
        return response
```

#### 3.3 임베딩 서비스

텍스트 임베딩을 위한 비즈니스 로직:

```python
class EmbeddingService:
    """텍스트 임베딩 서비스"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> Tuple[List[float], int]:
        """텍스트의 임베딩을 생성합니다."""
        # 요청 검증
        if not text or not text.strip():
            raise InvalidRequestError("임베딩할 텍스트는 비어 있을 수 없습니다")
        
        # 모델 매니저를 통해 임베딩 생성
        embeddings = await self.model_manager.get_embeddings(text, model_name=model)
        
        # 토큰 수 추정
        token_count = max(1, len(text) // 4)
        
        return embeddings, token_count
```

### 4. 모델 관리 계층

#### 4.1 통합 모델 매니저 (UnifiedModelManager)

다양한 모델 백엔드를 통합적으로 관리하는 클래스:

```python
class UnifiedModelManager:
    """통합 모델 관리 클래스"""
    
    def __init__(self):
        self.llm_manager = None
        self.transformers_manager = None
        self.active_backend: Optional[Literal["llama-cpp", "transformers"]] = None
        self.active_model: Optional[str] = None
        
        # 선택된 백엔드 또는 자동 감지
        self.backend = settings.MODEL_BACKEND
        
        # 필요한 백엔드 초기화
        if self.backend in ["llama-cpp", "auto"] and LLAMA_CPP_AVAILABLE:
            self.llm_manager = LLMManager()
        
        if self.backend in ["transformers", "auto"] and TRANSFORMERS_AVAILABLE:
            self.transformers_manager = TransformersManager()
    
    async def load_model(self, model_name: str, config: Optional[ModelConfig] = None, 
                         backend: Optional[Literal["llama-cpp", "transformers"]] = None) -> None:
        """모델을 로드합니다."""
        # 백엔드 결정 및 모델 로드 로직
        pass
    
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
        # 백엔드에 따라 적절한 매니저를 사용하여 채팅 완성 생성
        pass
```

#### 4.2 llama-cpp-python 백엔드 (LLMManager)

GGUF 형식의 모델을 관리하는 클래스:

```python
class LLMManager:
    """LLM 모델 관리 클래스"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.active_model: Optional[str] = None
        self.model_dir = Path(settings.MODEL_DIR)
        self.lock = asyncio.Lock()
        
        # llama-cpp-python 라이브러리 임포트
        from llama_cpp import Llama
        self.Llama = Llama
    
    async def load_model(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        """모델을 로드합니다."""
        # GGUF 모델 로드 로직
        pass
    
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
        # 채팅 메시지를 프롬프트로 변환
        prompt = self._format_chat_messages(messages)
        
        # llama-cpp-python을 사용하여 텍스트 생성
        pass
```

#### 4.3 Transformers 백엔드 (TransformersManager)

Hugging Face Transformers 모델을 관리하는 클래스:

```python
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
        
        # transformers 라이브러리 임포트
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
    
    async def load_model(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        """모델을 로드합니다."""
        # Transformers 모델 로드 로직
        pass
    
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
        # 채팅 메시지를 프롬프트로 변환
        prompt = self._format_chat_messages(messages, tokenizer)
        
        # Transformers를 사용하여 텍스트 생성
        pass
    
    def _format_chat_messages(self, messages: List[Dict[str, str]], tokenizer) -> str:
        """채팅 메시지를 Transformers 모델에 맞는 프롬프트 형식으로 변환합니다."""
        # 모델별 프롬프트 형식 적용 (Llama, Gemma, Mistral 등)
        pass
```

### 5. 핵심 구성 요소

#### 5.1 설정 관리 (app/config.py)

애플리케이션 전체 설정을 관리:

```python
class Settings(BaseSettings):
    # 기본 설정
    APP_NAME: str = "Local LLM API"
    API_VERSION: str = "v1"
    
    # 모델 유형
    MODEL_BACKEND: Literal["llama-cpp", "transformers", "auto"] = "auto"
    
    # GGUF 모델 설정
    MODEL_DIR: str = "./models"
    DEFAULT_MODEL: str = "llama3-8b-instruct.Q4_K_M.gguf"
    
    # Transformers 모델 설정
    TRANSFORMERS_DEFAULT_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # LLM 설정
    PRELOAD_MODEL: bool = True
    CONTEXT_SIZE: int = 4096
    TEMPERATURE: float = 0.7
    # ... 기타 설정
```

#### 5.2 로깅 시스템 (app/utils/logger.py)

구조화된 로깅 시스템:

```python
def setup_logger(name: str = None):
    """로거 설정"""
    # 로거 생성 및 설정
    logger = logging.getLogger(name or "app")
    logger.setLevel(LOG_LEVELS.get(settings.LOG_LEVEL, logging.INFO))
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    
    # 파일 핸들러 설정 (선택 사항)
    if settings.LOG_FILE:
        file_handler = TimedRotatingFileHandler(...)
        
    return logger
```

#### 5.3 예외 처리 (app/utils/exceptions.py)

사용자 정의 예외 클래스:

```python
class ModelNotLoadedError(Exception):
    """모델이 로드되지 않았을 때 발생하는 예외"""
    pass

class InferenceError(Exception):
    """모델 추론 중 오류가 발생했을 때 발생하는 예외"""
    pass

class InvalidRequestError(Exception):
    """유효하지 않은 요청에 대한 예외"""
    pass
```

## 주요 기능 및 특징

1. **다중 백엔드 지원**: GGUF 모델(llama-cpp-python)과 Hugging Face 모델(Transformers) 모두 지원
2. **통합 API**: 동일한 API로 다양한 백엔드의 모델 사용 가능
3. **자동 모델 감지**: 모델 파일 형식에 따라 적절한 백엔드 자동 선택
4. **모델별 프롬프팅**: 각 모델 계열에 맞는 최적의 프롬프트 형식 자동 적용
5. **비동기 처리**: FastAPI와 asyncio를 활용한 비동기 요청 처리
6. **GPU 가속**: CUDA를 통한 GPU 가속 지원
7. **OpenAI 호환 API**: OpenAI와 호환되는 API 형식 제공

## 설치 및 실행

### 환경 설정

```bash
# 저장소 클론
git clone https://github.com/yourusername/local-llm-api.git
cd local-llm-api

# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 모델 준비

```bash
# GGUF 모델 다운로드
mkdir -p models
wget -O models/llama3-8b-instruct.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf
```

### 실행

```bash
# 직접 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 또는 Docker Compose로 실행
docker-compose up -d
```

## API 사용 예시

### 채팅 완성

```bash
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
      {"role": "user", "content": "안녕하세요, 자기소개 부탁해요."}
    ],
    "model": "llama3-8b",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

### 텍스트 완성

```bash
curl -X POST "http://localhost:8000/api/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "다음은 한국의 전통 음식에 대한 설명입니다:",
    "model": "meta-llama/Llama-3-8B-Instruct",
    "max_tokens": 300,
    "temperature": 0.8
  }'
```

### 임베딩 생성

```bash
curl -X POST "http://localhost:8000/api/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "한국의 역사와 문화는 매우 풍부합니다.",
    "model": "mistral-7b"
  }'
```

## 결론

이 아키텍처는 로컬 환경에서 다양한 LLM 모델을 쉽게 활용할 수 있는 통합 API 서버를 제공합니다. GGUF 형식의 모델과 Hugging Face Transformers 모델을 모두 지원하여 사용자가 필요에 따라 적절한 모델을 선택할 수 있습니다. FastAPI를 기반으로 한 비동기 처리와 모듈식 설계로 확장성과 유지보수성이 뛰어납니다.

---
# 로컬 LLM API 서버 소스코드 구조

## 디렉토리 구조

```
local-llm-api/
├── app/                            # 메인 애플리케이션 패키지
│   ├── __init__.py
│   ├── main.py                     # FastAPI 애플리케이션 진입점
│   ├── config.py                   # 설정 관리
│   ├── core/                       # 핵심 기능 구현
│   │   ├── __init__.py
│   │   ├── llm_manager.py          # GGUF 모델 관리자 (llama-cpp-python)
│   │   ├── transformers_manager.py # Hugging Face 모델 관리자
│   │   ├── unified_model_manager.py # 통합 모델 관리자
│   │   └── model_config.py         # 모델 설정 클래스
│   ├── api/                        # API 라우트 정의
│   │   ├── __init__.py
│   │   ├── dependencies.py         # API 종속성 (인증 등)
│   │   └── routes/                 # API 라우트 모듈
│   │       ├── __init__.py
│   │       ├── chat.py             # 채팅 API 라우트
│   │       ├── completion.py       # 완성 API 라우트
│   │       ├── embeddings.py       # 임베딩 API 라우트
│   │       └── health.py           # 상태 확인 및 모델 관리 API 라우트
│   ├── services/                   # 서비스 계층
│   │   ├── __init__.py
│   │   ├── chat_service.py         # 채팅 완성 서비스
│   │   ├── completion_service.py   # 텍스트 완성 서비스
│   │   └── embedding_service.py    # 텍스트 임베딩 서비스
│   ├── models/                     # Pydantic 모델 정의
│   │   ├── __init__.py
│   │   └── request_models.py       # API 요청 및 응답 모델
│   └── utils/                      # 유틸리티 모듈
│       ├── __init__.py
│       ├── logger.py               # 로깅 유틸리티
│       └── exceptions.py           # 사용자 정의 예외 클래스
├── tests/                          # 테스트 코드
│   ├── __init__.py
│   ├── test_chat.py                # 채팅 API 테스트
│   ├── test_completion.py          # 완성 API 테스트
│   └── test_embeddings.py          # 임베딩 API 테스트
├── models/                         # 모델 파일 저장 디렉토리
│   └── .gitkeep
├── logs/                           # 로그 파일 저장 디렉토리
│   └── .gitkeep
├── nginx/                          # Nginx 설정 (선택 사항)
│   └── nginx.conf
├── .env                            # 환경 변수 파일
├── .gitignore                      # Git 무시 파일 목록
├── requirements.txt                # Python 의존성 파일
├── Dockerfile                      # Docker 빌드 파일
├── docker-compose.yml              # Docker Compose 설정
└── README.md                       # 프로젝트 설명서
```

## 핵심 파일 설명

### 1. 메인 애플리케이션 (app/main.py)

FastAPI 애플리케이션의 진입점으로, 서버 초기화, 미들웨어 설정, 라우터 등록 등을 담당합니다.

```python
from fastapi import FastAPI
from app.core.unified_model_manager import UnifiedModelManager
from app.api.routes import chat, completion, embeddings, health
from app.utils.logger import setup_logger

# FastAPI 앱 초기화
app = FastAPI(title="Local LLM API", ...)

# 미들웨어, 예외 핸들러 등록
@app.middleware("http")
async def log_requests(request, call_next): ...

# 라우터 등록
app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
...

# 서버 시작 이벤트
@app.on_event("startup")
async def startup_event():
    model_manager = UnifiedModelManager()
    app.state.model_manager = model_manager
    ...
```

### 2. 설정 관리 (app/config.py)

환경 변수와 기본값을 관리하는 Pydantic 기반 설정 클래스를 정의합니다.

```python
from pydantic import BaseSettings
from typing import List, Dict, Optional, Any, Literal

class Settings(BaseSettings):
    # 기본 설정
    APP_NAME: str = "Local LLM API"
    API_VERSION: str = "v1"
    
    # 모델 유형
    MODEL_BACKEND: Literal["llama-cpp", "transformers", "auto"] = "auto"
    
    # GGUF 모델 설정
    MODEL_DIR: str = "./models"
    DEFAULT_MODEL: str = "llama3-8b-instruct.Q4_K_M.gguf"
    
    # Transformers 모델 설정
    TRANSFORMERS_DEFAULT_MODEL: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ...

# 설정 인스턴스 생성
settings = Settings()
```

### 3. 통합 모델 관리자 (app/core/unified_model_manager.py)

여러 백엔드(GGUF, Transformers)를 통합적으로 관리하는 클래스를 정의합니다.

```python
class UnifiedModelManager:
    """통합 모델 관리 클래스"""
    
    def __init__(self):
        self.llm_manager = None
        self.transformers_manager = None
        self.active_backend = None
        self.active_model = None
        
        # 백엔드 초기화
        if self.backend in ["llama-cpp", "auto"] and LLAMA_CPP_AVAILABLE:
            self.llm_manager = LLMManager()
        
        if self.backend in ["transformers", "auto"] and TRANSFORMERS_AVAILABLE:
            self.transformers_manager = TransformersManager()
    
    async def load_model(self, model_name, config=None, backend=None): ...
    
    async def generate_completion(self, prompt, model_name=None, ...): ...
    
    async def generate_chat_completion(self, messages, model_name=None, ...): ...
    
    async def get_embeddings(self, text, model_name=None): ...
```

### 4. GGUF 모델 관리자 (app/core/llm_manager.py)

llama-cpp-python을 사용하여 GGUF 형식의 모델을 관리하는 클래스를 정의합니다.

```python
class LLMManager:
    """LLM 모델 관리 클래스"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.active_model = None
        self.model_dir = Path(settings.MODEL_DIR)
        
        # llama-cpp-python 임포트
        from llama_cpp import Llama
        self.Llama = Llama
    
    async def load_model(self, model_name, config=None): ...
    
    async def generate_completion(self, prompt, model_name=None, ...): ...
    
    def _format_chat_messages(self, messages): ...
```

### 5. Transformers 모델 관리자 (app/core/transformers_manager.py)

Hugging Face Transformers를 사용하여 모델을 관리하는 클래스를 정의합니다.

```python
class TransformersManager:
    """Hugging Face Transformers 모델 관리 클래스"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_configs = {}
        self.active_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # transformers 임포트
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
    
    async def load_model(self, model_name, config=None): ...
    
    async def generate_completion(self, prompt, model_name=None, ...): ...
    
    def _format_chat_messages(self, messages, tokenizer): ...
```

### 6. API 라우트 (app/api/routes/)

#### 6.1 채팅 API (app/api/routes/chat.py)

채팅 완성 API 엔드포인트를 정의합니다.

```python
@router.post("/chat/completions", response_model=Dict[str, Any])
async def create_chat_completion(
    request: Request,
    chat_request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    # 모델 매니저 가져오기
    model_manager = request.app.state.model_manager
    
    # 채팅 서비스 생성
    chat_service = ChatService(model_manager)
    
    # 채팅 완성 생성
    response = await chat_service.generate_chat_completion(...)
    
    return response
```

#### 6.2 완성 API (app/api/routes/completion.py)

텍스트 완성 API 엔드포인트를 정의합니다.

```python
@router.post("/completions", response_model=Dict[str, Any])
async def create_completion(
    request: Request,
    completion_request: CompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    # 모델 매니저 가져오기
    model_manager = request.app.state.model_manager
    
    # 완성 서비스 생성
    completion_service = CompletionService(model_manager)
    
    # 텍스트 완성 생성
    response = await completion_service.generate_completion(...)
    
    return response
```

#### 6.3 임베딩 API (app/api/routes/embeddings.py)

텍스트 임베딩 API 엔드포인트를 정의합니다.

```python
@router.post("/embeddings", response_model=Dict[str, Any])
async def create_embeddings(
    request: Request,
    embedding_request: EmbeddingRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    # 모델 매니저 가져오기
    model_manager = request.app.state.model_manager
    
    # 임베딩 서비스 생성
    embedding_service = EmbeddingService(model_manager)
    
    # 임베딩 생성
    embeddings, token_count = await embedding_service.generate_embedding(...)
    
    return {
        "object": "list",
        "data": [{"embedding": embeddings, "index": 0, "object": "embedding"}],
        "model": embedding_request.model or model_manager.active_model,
        "usage": {"prompt_tokens": token_count, "total_tokens": token_count}
    }
```

#### 6.4 상태 확인 및 모델 관리 API (app/api/routes/health.py)

서버 상태 확인 및 모델 관리 API 엔드포인트를 정의합니다.

```python
@router.get("/health", response_model=Dict[str, Any])
async def health_check(request: Request):
    # 시스템 정보 수집
    system_info = {...}
    
    # 모델 매니저 상태 확인
    if hasattr(request.app.state, "model_manager"):
        model_manager = request.app.state.model_manager
        model_info = {...}
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "system": system_info,
        "model": model_info
    }

@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(request: Request):
    # 모델 매니저 가져오기
    model_manager = request.app.state.model_manager
    
    # 모델 목록 가져오기
    models = await model_manager.list_available_models()
    
    return models

@router.post("/models/{model_name}/load", response_model=Dict[str, Any])
async def load_model(request: Request, model_name: str):
    # 모델 매니저 가져오기
    model_manager = request.app.state.model_manager
    
    # 모델 로드
    await model_manager.load_model(model_name)
    
    return {
        "status": "success",
        "message": f"모델 '{model_name}' 로드 성공",
        "model": model_name
    }
```

### 7. 서비스 계층 (app/services/)

#### 7.1 채팅 서비스 (app/services/chat_service.py)

채팅 완성을 위한 비즈니스 로직을 구현합니다.

```python
class ChatService:
    """채팅 완성 서비스"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    async def generate_chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """채팅 완성을 생성합니다."""
        # 요청 검증
        if not messages:
            raise InvalidRequestError("메시지 목록은 비어 있을 수 없습니다")
        
        # 메시지 형식 변환
        formatted_messages = [msg.dict() for msg in messages]
        
        # 모델 매니저를 통해 채팅 완성 생성
        response = await self.model_manager.generate_chat_completion(...)
        
        return response
```

#### 7.2 완성 서비스 (app/services/completion_service.py)

텍스트 완성을 위한 비즈니스 로직을 구현합니다.

```python
class CompletionService:
    """텍스트 완성 서비스"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    async def generate_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """텍스트 완성을 생성합니다."""
        # 요청 검증
        if not prompt or not prompt.strip():
            raise InvalidRequestError("프롬프트는 비어 있을 수 없습니다")
        
        # 모델 매니저를 통해 텍스트 완성 생성
        response = await self.model_manager.generate_completion(...)
        
        return response
```

#### 7.3 임베딩 서비스 (app/services/embedding_service.py)

텍스트 임베딩을 위한 비즈니스 로직을 구현합니다.

```python
class EmbeddingService:
    """텍스트 임베딩 서비스"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> Tuple[List[float], int]:
        """텍스트의 임베딩을 생성합니다."""
        # 요청 검증
        if not text or not text.strip():
            raise InvalidRequestError("임베딩할 텍스트는 비어 있을 수 없습니다")
        
        # 모델 매니저를 통해 임베딩 생성
        embeddings = await self.model_manager.get_embeddings(text, model_name=model)
        
        # 토큰 수 추정
        token_count = max(1, len(text) // 4)
        
        return embeddings, token_count
```

### 8. 모델 정의 (app/models/request_models.py)

API 요청 및 응답을 위한 Pydantic 모델을 정의합니다.

```python
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
```

### 9. 유틸리티 (app/utils/)

#### 9.1 로깅 유틸리티 (app/utils/logger.py)

로깅 시스템을 설정합니다.

```python
def setup_logger(name: str = None):
    """로거 설정"""
    # 로거 생성 및 설정
    logger = logging.getLogger(name or "app")
    logger.setLevel(LOG_LEVELS.get(settings.LOG_LEVEL, logging.INFO))
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)
    
    # 파일 핸들러 설정 (선택 사항)
    if settings.LOG_FILE:
        file_handler = TimedRotatingFileHandler(...)
        logger.addHandler(file_handler)
    
    return logger
```

#### 9.2 예외 클래스 (app/utils/exceptions.py)

사용자 정의 예외 클래스를 정의합니다.

```python
class ModelNotLoadedError(Exception):
    """모델이 로드되지 않았을 때 발생하는 예외"""
    def __init__(self, message="모델이 로드되지 않았습니다"):
        self.message = message
        super().__init__(self.message)

class InferenceError(Exception):
    """모델 추론 중 오류가 발생했을 때 발생하는 예외"""
    def __init__(self, message="모델 추론 중 오류가 발생했습니다"):
        self.message = message
        super().__init__(self.message)

class InvalidRequestError(Exception):
    """유효하지 않은 요청에 대한 예외"""
    def __init__(self, message="유효하지 않은 요청입니다"):
        self.message = message
        super().__init__(self.message)
```

### 10. 배포 파일

#### 10.1 Dockerfile

Docker 이미지 빌드를 위한 설정입니다.

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul

# 기본 패키지 및 종속성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    curl \
    wget \
    git \
    cmake \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 생성
WORKDIR /app

# 모델 및 로그 디렉토리 생성
RUN mkdir -p /app/models /app/logs

# 애플리케이션 종속성 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 설정
EXPOSE 8000

# 엔트리포인트 설정
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 10.2 docker-compose.yml

Docker Compose 설정입니다.

```yaml
version: '3.8'

services:
  llm-api:
    build: 
      context: .
      dockerfile: Dockerfile
    container_name: local-llm-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - MODEL_DIR=/app/models
      - DEFAULT_MODEL=llama3-8b-instruct.Q4_K_M.gguf
      - GPU_LAYERS=-1
      - PRELOAD_MODEL=true
      - MODEL_BACKEND=auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: llm-api-nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - llm-api
    restart: unless-stopped

networks:
  default:
    name: llm-network
```

## 요약

이 소스코드 구조는 로컬 LLM API 서버를 위한 모듈식, 계층적인 아키텍처를 제공합니다. FastAPI를 기반으로 한 웹 서버, 다양한 모델 백엔드를 지원하는 통합 모델 관리자, 그리고 채팅, 완성, 임베딩 API를 제공하는 서비스 계층으로 구성되어 있습니다. 이 구조는 확장성과 유지보수성을 고려하여 설계되었으며, Docker 지원을 통해 손쉬운 배포가 가능합니다.