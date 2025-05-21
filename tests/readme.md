# vLLM 백엔드 테스트 설정 가이드

이 가이드는 vLLM 백엔드가 추가된 로컬 LLM API 서버를 설정하고 테스트하는 방법을 설명합니다.

## 1. 환경 준비

vLLM을 사용하기 위해서는 CUDA 호환 GPU가 필요합니다. 최소 요구사항은 다음과 같습니다:

- NVIDIA GPU (최소 8GB VRAM, 권장 16GB 이상)
- CUDA 11.8 이상
- Python 3.9+
- torch 2.1.0+

### 1.1 CUDA 설치

Ubuntu 시스템의 경우:

```bash
# CUDA 툴킷 설치
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

### 1.2 Python 가상환경 생성

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

# vLLM 및 기타 의존성 설치
pip install -r requirements.txt

# PyTorch가 GPU를 인식하는지 확인
python -c "import torch; print('CUDA 가능:', torch.cuda.is_available(), ', 장치 수:', torch.cuda.device_count())"
```

만약 PyTorch가 GPU를 인식하지 못한다면 호환되는 버전의 PyTorch를 수동으로 설치하세요:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

## 3. 설정 파일 수정

`.env` 파일을 생성하거나 수정하여 vLLM 및 기타 설정을 지정합니다:

```bash
# .env 파일 생성 또는 수정
cat > .env << EOL
# 앱 기본 설정
APP_NAME=Local LLM API
API_VERSION=v1
DEBUG=True

# 모델 백엔드 설정
# "llama-cpp", "transformers", "vllm", "auto" 중 선택
MODEL_BACKEND=vllm

# vLLM 모델 설정
VLLM_DEFAULT_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# LLM 설정
MAX_TOKENS=1024
TEMPERATURE=0.7
EOL
```

## 4. 테스트용 모델 준비

vLLM 테스트를 위해 가벼운 모델을 사용할 수 있습니다:

```bash
# 소형 모델 사용 (테스트용)
# TinyLlama는 약 1.1B 파라미터 모델로 테스트용으로 적합
# vLLM은 Hugging Face 모델을 자동으로 다운로드합니다
```

더 큰 모델을 테스트하려면 설정 파일에서 모델 이름을 변경하세요:

```python
# config.py 파일에서 vLLM 모델 설정
VLLM_DEFAULT_MODEL: str = "meta-llama/Llama-3-8B-Instruct"  # 또는 다른 모델
```

## 5. 서버 실행

개발 모드에서 서버를 실행합니다:

```bash
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

### 6.1 상태 확인 및 모델 목록

```bash
# 서버 상태 확인
curl -X GET "http://localhost:8000/health"

# 사용 가능한 모델 목록
curl -X GET "http://localhost:8000/models"
```

### 6.2 vLLM 모델 로드

```bash
# vLLM 모델 로드 (기본 모델이 자동으로 로드되지 않은 경우)
curl -X POST "http://localhost:8000/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0/load"

# 또는 모델 별칭 사용
curl -X POST "http://localhost:8000/models/llama3-8b/load"
```

### 6.3 채팅 API 호출

```bash
# 채팅 완성 API 호출
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
      {"role": "user", "content": "안녕하세요! vLLM을 사용한 추론은 어떤 장점이 있나요?"}
    ],
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### 6.4 텍스트 완성 API 호출

```bash
# 텍스트 완성 API 호출
curl -X POST "http://localhost:8000/api/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "vLLM은 다음과 같은 장점이 있습니다:",
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_tokens": 150,
    "temperature": 0.8
  }'
```

### 6.5 임베딩 API 호출

```bash
# 임베딩 API 호출 (vLLM 모델에서 지원하는 경우)
curl -X POST "http://localhost:8000/api/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "vLLM은 대규모 언어 모델을 위한 고성능 추론 엔진입니다.",
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  }'
```

## 7. 성능 테스트

vLLM의 성능을 테스트하기 위해 벤치마크 스크립트를 실행할 수 있습니다:

```bash
# 간단한 벤치마크 스크립트 생성
cat > benchmark.py << EOF
import asyncio
import time
import httpx

async def send_request(client, i):
    start_time = time.time()
    response = await client.post(
        "http://localhost:8000/api/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다."},
                {"role": "user", "content": f"간단한 벤치마크 테스트입니다. 테스트 번호: {i}"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
    )
    end_time = time.time()
    return end_time - start_time, response.status_code

async def main():
    num_requests = 10  # 동시 요청 수
    async with httpx.AsyncClient() as client:
        tasks = [send_request(client, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
    
    # 결과 분석
    times, statuses = zip(*results)
    print(f"총 요청: {num_requests}")
    print(f"평균 응답 시간: {sum(times)/len(times):.3f}초")
    print(f"최소 응답 시간: {min(times):.3f}초")
    print(f"최대 응답 시간: {max(times):.3f}초")
    print(f"성공 요청: {statuses.count(200)}")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# 벤치마크 실행
python benchmark.py
```

## 8. 문제 해결

### 8.1 CUDA 문제

```bash
# CUDA 버전 및 드라이버 확인
nvidia-smi
nvcc --version

# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 8.2 vLLM 오류

vLLM 초기화 오류가 발생하면 다음을 확인하세요:

```bash
# vLLM 버전 확인
pip show vllm

# 호환성 확인
python -c "import vllm; print(vllm.__version__)"
```

### 8.3 메모리 부족 문제

대형 모델에서 메모리 부족 오류가 발생하면:

1. 더 작은 모델로 테스트
2. 텐서 병렬화 활성화 (여러 GPU에 모델 분산)
3. 양자화 옵션 사용

```python
# config.py에서 양자화 및 텐서 병렬화 설정
VLLM_MODELS: Dict[str, Dict[str, Any]] = {
    "llama3-70b": {
        "model_id": "meta-llama/Llama-3-70B-Instruct",
        "tensor_parallel_size": 2,  # GPU 2개에 분산
        "quantization": "awq"  # 양자화 사용
    }
}
```

## 9. vLLM과 다른 백엔드 간 벤치마크 비교

서로 다른 백엔드 간의 성능을 비교하기 위해 다음과 같은 벤치마크를 실행할 수 있습니다:

```bash
# 벤치마크 비교 스크립트 생성
cat > benchmark_compare.py << EOF
import asyncio
import time
import httpx
import pandas as pd
import matplotlib.pyplot as plt

async def send_request(client, backend, model, prompt):
    start_time = time.time()
    response = await client.post(
        "http://localhost:8000/api/v1/completions",
        json={
            "prompt": prompt,
            "model": model,
            "max_tokens": 50,
            "temperature": 0.7
        }
    )
    end_time = time.time()
    return {
        "backend": backend,
        "model": model,
        "time": end_time - start_time,
        "status": response.status_code
    }

async def test_backend(client, backend, model, prompt, num_requests=5):
    print(f"테스트 중: {backend} - {model}")
    tasks = [send_request(client, backend, model, prompt) for _ in range(num_requests)]
    return await asyncio.gather(*tasks)

async def main():
    # 테스트 설정
    tests = [
        {"backend": "vllm", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        {"backend": "transformers", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
        {"backend": "llama-cpp", "model": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"}
    ]
    prompt = "인공지능 기술의 미래에 대해 간략히 설명해주세요."
    num_requests = 5  # 각 백엔드당 요청 수
    
    results = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        for test in tests:
            try:
                test_results = await test_backend(
                    client, 
                    test["backend"], 
                    test["model"], 
                    prompt, 
                    num_requests
                )
                results.extend(test_results)
            except Exception as e:
                print(f"오류 발생: {test['backend']} - {e}")
    
    # 결과 분석
    if results:
        df = pd.DataFrame(results)
        print("\n결과 요약:")
        summary = df.groupby('backend')['time'].agg(['mean', 'min', 'max', 'count'])
        print(summary)
        
        # 차트 생성
        plt.figure(figsize=(10, 6))
        df.boxplot(column='time', by='backend')
        plt.title('백엔드별 추론 시간 비교')
        plt.ylabel('응답 시간 (초)')
        plt.savefig('benchmark_results.png')
        print("\n차트가 'benchmark_results.png'로 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# 필요한 패키지 설치
pip install pandas matplotlib

# 벤치마크 비교 실행
python benchmark_compare.py
```

## 10. 프로덕션 배포 고려사항

### 10.1 리소스 요구사항

vLLM은 고성능 GPU 추론에 최적화되어 있지만, 다음 사항을 고려해야 합니다:

- **메모리 요구사항**: 대형 모델(예: 70B)은 최소 80GB 이상의 VRAM 필요
- **다중 GPU**: 텐서 병렬화를 통해 여러 GPU에 모델 분산 가능
- **디스크 공간**: 모델 가중치 저장을 위한 충분한 공간 확보(최소 모델 크기의 2배)

### 10.2 로드 밸런싱

대규모 배포의 경우:

```bash
# nginx 구성 예시
cat > nginx/nginx.conf << EOF
http {
    upstream llm_servers {
        server llm-server1:8000;
        server llm-server2:8000;
        server llm-server3:8000;
    }
    
    server {
        listen 80;
        
        location / {
            proxy_pass http://llm_servers;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_read_timeout 300s;  # 긴 요청 처리를 위한 타임아웃 증가
        }
    }
}
EOF
```

### 10.3 모니터링

Prometheus 및 Grafana를 사용하여 서버 상태 모니터링:

```yaml
# docker-compose.yml에 추가
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - 9090:9090
      
  grafana:
    image: grafana/grafana
    depends_on:
      - prometheus
    ports:
      - 3000:3000
```

## 11. 결론

vLLM 백엔드는 대규모 언어 모델의 추론 성능을 크게 향상시킵니다. 특히 동시에 여러 요청을 처리하는 경우, PagedAttention 기술을 통해 메모리 효율성과 처리량을 모두 개선합니다. 이 가이드를 통해 vLLM 백엔드를 테스트하고, 다른 백엔드와 비교하여 사용 사례에 가장 적합한 백엔드를 선택할 수 있습니다.