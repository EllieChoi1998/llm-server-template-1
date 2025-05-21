# 로컬 LLM API v1 테스트 설정 가이드

이 가이드는 llama-cpp-python과 Hugging Face Transformers 백엔드를 지원하는 로컬 LLM API 서버(v1)를 설정하고 테스트하는 방법을 설명합니다.

## 1. 환경 준비

### 1.1 시스템 요구사항

- **CPU 전용 실행**: 최소 8GB RAM, 권장 16GB 이상
- **GPU 가속**: NVIDIA GPU (최소 4GB VRAM, 권장 8GB 이상)
- **CUDA**: 11.7 이상 (GPU 가속 사용 시)
- **Python**: 3.9+

### 1.2 CUDA 설치 (GPU 가속 사용 시)

Ubuntu 시스템의 경우:

```bash
# CUDA 툴킷 설치
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### 1.3 Python 가상환경 생성

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

## 2. 의존성 설치

저장소를 클론하고 필요한 패키지를 설치합니다:

```bash
# 저장소 클론
git clone https://github.com/yourusername/local-llm-api.git
cd local-llm-api

# 의존성 설치
pip install -r requirements.txt

# GPU 지원 확인 (GPU 가속 사용 시)
python -c "import torch; print('CUDA 가능:', torch.cuda.is_available(), ', 장치 수:', torch.cuda.device_count())"
```

만약 PyTorch가 GPU를 인식하지 못한다면 호환되는 버전의 PyTorch를 수동으로 설치하세요:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## 3. 모델 다운로드

### 3.1 GGUF 모델 (llama-cpp-python 백엔드용)

```bash
# models 디렉토리 생성
mkdir -p models

# TinyLlama 모델 다운로드 (예시)
wget -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# 또는 더 좋은 품질의 llama3-8b 모델 (용량이 더 큼)
# wget -O models/llama3-8b-instruct.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf
```

### 3.2 Hugging Face 모델 (Transformers 백엔드용)

```bash
# HuggingFace 모델은 사용 시 자동으로 다운로드됩니다.
# TinyLlama 모델은 약 1.1B 파라미터 모델로 테스트용으로 적합합니다.
```

## 4. 설정 파일 수정

`.env` 파일을 생성하거나 수정하여 필요한 설정을 지정합니다:

```bash
# .env 파일 생성 또는 수정
cat > .env << EOL
# 앱 기본 설정
APP_NAME=Local LLM API
API_VERSION=v1
DEBUG=True

# 모델 백엔드 설정
# "llama-cpp", "transformers", "auto" 중 선택
MODEL_BACKEND=auto

# GGUF 모델 설정
MODEL_DIR=./models
DEFAULT_MODEL=tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
PRELOAD_MODEL=True

# Transformers 모델 설정
TRANSFORMERS_DEFAULT_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# LLM 설정
CONTEXT_SIZE=2048
GPU_LAYERS=-1
MAX_TOKENS=1024
TEMPERATURE=0.7
TOP_P=0.95
TOP_K=40

# 로깅 설정
LOG_LEVEL=INFO
LOG_FILE=./logs/api.log
EOL
```

## 5. 서버 실행

개발 모드에서 서버를 실행합니다:

```bash
# 로그 디렉토리 생성
mkdir -p logs

# 직접 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

또는 Docker 컨테이너로 실행:

```bash
# Docker Compose로 실행
docker-compose up -d
```

## 6. API 테스트

서버가 실행 중인 상태에서 다음 테스트를 시도할 수 있습니다:

### 6.1 API 문서 확인

브라우저에서 다음 URL을 열어 API 문서를 확인합니다:
- http://localhost:8000/docs

### 6.2 상태 확인 및 모델 목록

```bash
# 서버 상태 확인
curl -X GET "http://localhost:8000/health"

# 사용 가능한 모델 목록
curl -X GET "http://localhost:8000/models"
```

### 6.3 모델 로드 (필요한 경우)

```bash
# GGUF 모델 로드 (기본 모델이 자동으로 로드되지 않은 경우)
curl -X POST "http://localhost:8000/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf/load"

# Transformers 모델 로드
curl -X POST "http://localhost:8000/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0/load"
```

### 6.4 채팅 API 테스트

#### GGUF 모델 사용

```bash
# GGUF 모델로 채팅 완성 API 호출
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
      {"role": "user", "content": "안녕하세요! 인공지능에 대해 간단히 설명해주세요."}
    ],
    "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

#### Transformers 모델 사용

```bash
# Transformers 모델로 채팅 완성 API 호출
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
      {"role": "user", "content": "안녕하세요! 인공지능에 대해 간단히 설명해주세요."}
    ],
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### 6.5 텍스트 완성 API 테스트

```bash
# 텍스트 완성 API 호출
curl -X POST "http://localhost:8000/api/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "인공지능의 주요 응용 분야는 다음과 같습니다:",
    "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "max_tokens": 150,
    "temperature": 0.8
  }'
```

### 6.6 임베딩 API 테스트

```bash
# 임베딩 API 호출
curl -X POST "http://localhost:8000/api/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "인공지능은 현대 기술의 중요한 부분입니다.",
    "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  }'
```

## 7. 성능 테스트

### 7.1 간단한 벤치마크 스크립트 작성

```bash
# 간단한 벤치마크 스크립트 생성
cat > benchmark_v1.py << EOF
import asyncio
import time
import httpx

async def send_request(client, model, i):
    start_time = time.time()
    response = await client.post(
        "http://localhost:8000/api/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
                {"role": "user", "content": f"간단한 벤치마크 테스트입니다. 테스트 번호: {i}"}
            ],
            "model": model,
            "max_tokens": 50,
            "temperature": 0.7
        }
    )
    end_time = time.time()
    return end_time - start_time, response.status_code

async def test_model(client, model, num_requests=3):
    print(f"\n테스트 모델: {model}")
    tasks = [send_request(client, model, i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    times, statuses = zip(*results)
    print(f"총 요청: {num_requests}")
    print(f"평균 응답 시간: {sum(times)/len(times):.3f}초")
    print(f"최소 응답 시간: {min(times):.3f}초")
    print(f"최대 응답 시간: {max(times):.3f}초")
    print(f"성공 요청: {statuses.count(200)}")
    return times

async def main():
    # 테스트할 모델
    models = [
        "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",  # GGUF 모델
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"     # Transformers 모델
    ]
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for model in models:
            try:
                await test_model(client, model)
            except Exception as e:
                print(f"오류 발생: {model} - {e}")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# 벤치마크 실행
python benchmark_v1.py
```

## 8. 문제 해결

### 8.1 GGUF 모델 문제

모델 로드 중 오류가 발생하면 다음을 확인하세요:

```bash
# GGUF 모델이 올바른 위치에 있는지 확인
ls -la models/

# 모델 파일이 손상되지 않았는지 확인 (파일 크기 확인)
du -h models/*.gguf
```

### 8.2 Transformers 모델 문제

```bash
# Transformers 버전 확인
pip show transformers

# 모델 캐시 경로 확인
python -c "from huggingface_hub import HfFolder; print(HfFolder.get_cache_dir())"

# 캐시 지우기 (문제 발생 시)
rm -rf ~/.cache/huggingface/
```

### 8.3 GPU 관련 문제

```bash
# GPU 상태 확인
nvidia-smi

# 메모리 사용량 모니터링
watch -n 1 "nvidia-smi"

# GPU 메모리 정리 (필요 시)
python -c "import torch; torch.cuda.empty_cache()"
```

### 8.4 일반적인 오류 해결

- **메모리 부족**: 더 작은 모델을 사용하거나 양자화된 모델로 전환
- **모델 로드 시간이 너무 긴 경우**: 처음 로드 후 모델을 언로드하지 않도록 설정
- **응답이 느린 경우**: `GPU_LAYERS` 설정을 확인하거나 더 작은 모델 사용

## 9. 백엔드 간 차이점

### 9.1 llama-cpp-python vs Transformers

| 특성 | llama-cpp-python (GGUF) | Transformers |
|------|------------------------|--------------|
| 파일 형식 | GGUF | Hugging Face 모델 |
| 메모리 사용량 | 더 적음 | 더 많음 |
| 로딩 속도 | 더 빠름 | 더 느림 |
| 추론 속도 | CPU에서 최적화됨 | GPU에서 최적화됨 |
| 양자화 | 내장 지원 | 추가 라이브러리 필요 |
| 모델 가용성 | 제한적 | 매우 다양함 |

### 9.2 백엔드 선택 가이드라인

- **CPU 전용 환경**: llama-cpp-python (GGUF) 권장
- **GPU 환경, 다양한 모델 필요**: Transformers 권장
- **소형 모델 (< 7B)**: 두 백엔드 모두 효율적
- **대형 모델 (> 7B)**: GPU가 있는 경우 Transformers 권장

## 10. 결론

로컬 LLM API v1은 llama-cpp-python과 Hugging Face Transformers 두 백엔드를 지원하여 다양한 환경과 모델에 유연하게 대응할 수 있습니다. CPU 전용 환경에서는 GGUF 모델을, GPU 환경에서는 Transformers 모델을 활용하여 최적의 성능을 얻을 수 있습니다. 이 테스트 가이드를 통해 각 백엔드의 특성을 파악하고 사용 사례에 가장 적합한 백엔드를 선택할 수 있습니다.