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