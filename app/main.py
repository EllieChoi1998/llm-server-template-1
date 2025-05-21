from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from app.api.routes import chat, completion, embeddings, health
from app.core.unified_model_manager import UnifiedModelManager
from app.utils.exceptions import ModelNotLoadedError, InferenceError
from app.config import settings
from app.utils.logger import setup_logger

# 로거 설정
logger = setup_logger("app")

# FastAPI 앱 생성
app = FastAPI(
    title="Local LLM API",
    description="로컬 LLM을 위한 FastAPI 서버",
    version="0.1.0",
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 미들웨어 - 요청 처리 시간 로깅
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request path: {request.url.path} - Time: {process_time:.3f}s")
    return response

# 예외 핸들러
@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc)},
    )

@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# 라우터 등록
app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(completion.router, prefix="/api/v1", tags=["completion"])
app.include_router(embeddings.router, prefix="/api/v1", tags=["embeddings"])

# 애플리케이션 시작 이벤트
@app.on_event("startup")
async def startup_event():
    logger.info("서버 시작 중...")
    
    # 통합 모델 매니저 초기화
    try:
        model_manager = UnifiedModelManager()
        app.state.model_manager = model_manager
        logger.info(f"모델 매니저 초기화 성공 (활성 백엔드: {model_manager.active_backend})")
        
        # 모델 사전 로드 (비동기로 로드를 시작하고 실제 요청은 준비 상태를 확인함)
        if settings.PRELOAD_MODEL:
            if model_manager.active_backend == "llama-cpp":
                default_model = settings.DEFAULT_MODEL
                logger.info(f"GGUF 모델 사전 로드 시작: {default_model}")
                await model_manager.load_model(default_model)
            elif model_manager.active_backend == "transformers":
                default_model = settings.TRANSFORMERS_DEFAULT_MODEL
                logger.info(f"Transformers 모델 사전 로드 시작: {default_model}")
                await model_manager.load_model(default_model)
            
            logger.info(f"모델 로드 완료: {model_manager.active_model}")
    except Exception as e:
        logger.error(f"모델 매니저 초기화 중 오류: {str(e)}")
        # 실패해도 서버는 시작할 수 있도록 함 (모델 로드는 API 요청 시 시도 가능)
    
    logger.info("서버 준비 완료")

# 애플리케이션 종료 이벤트
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("서버 종료 중...")
    # 모델 리소스 해제
    if hasattr(app.state, "model_manager"):
        await app.state.model_manager.unload_all_models()
    logger.info("서버 종료 완료")

# 직접 실행 시
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)