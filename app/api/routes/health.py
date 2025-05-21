from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any, List
import logging
import time
import os
import psutil
import platform
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health", response_model=Dict[str, Any])
async def health_check(request: Request):
    """서버 상태 확인 엔드포인트"""
    start_time = time.time()
    
    # 기본 상태 확인
    status = "ok"
    
    try:
        # 시스템 정보 수집
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "cpu_usage": psutil.cpu_percent(interval=0.1)
        }
        
        # 모델 매니저 상태 확인
        model_info = {}
        if hasattr(request.app.state, "model_manager"):
            model_manager = request.app.state.model_manager
            
            # 백엔드 정보
            model_info["backend"] = model_manager.active_backend
            
            # 활성 모델 정보
            active_model = model_manager.active_model
            model_info["active_model"] = active_model
            
            # 사용 가능한 모델 목록
            if not hasattr(request.app.state, "available_models_cached") or \
               (datetime.now() - request.app.state.available_models_cached_time).total_seconds() > 300:
                # 5분마다 캐시 갱신
                available_models = await model_manager.list_available_models()
                request.app.state.available_models_cached = available_models
                request.app.state.available_models_cached_time = datetime.now()
            else:
                available_models = request.app.state.available_models_cached
                
            model_info["available_models"] = available_models
        else:
            model_info["status"] = "not_initialized"
            status = "warning"
        
        # 응답 시간 계산
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - psutil.boot_time(),
            "response_time": response_time,
            "system": system_info,
            "model": model_info
        }
    
    except Exception as e:
        logger.error(f"상태 확인 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"상태 확인 중 오류 발생: {str(e)}"
        )


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(request: Request):
    """사용 가능한 모델 목록 반환 엔드포인트"""
    try:
        # 모델 매니저 가져오기
        if not hasattr(request.app.state, "model_manager"):
            raise HTTPException(
                status_code=503,
                detail="모델 매니저가 초기화되지 않았습니다"
            )
        
        model_manager = request.app.state.model_manager
        
        # 모델 목록 가져오기
        models = await model_manager.list_available_models()
        
        return models
    
    except Exception as e:
        logger.error(f"모델 목록 조회 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"모델 목록 조회 중 오류 발생: {str(e)}"
        )


@router.post("/models/{model_name}/load", response_model=Dict[str, Any])
async def load_model(request: Request, model_name: str):
    """모델 로드 엔드포인트"""
    try:
        # 모델 매니저 가져오기
        if not hasattr(request.app.state, "model_manager"):
            raise HTTPException(
                status_code=503,
                detail="모델 매니저가 초기화되지 않았습니다"
            )
        
        model_manager = request.app.state.model_manager
        
        # 모델 로드
        start_time = time.time()
        await model_manager.load_model(model_name)
        end_time = time.time()
        
        return {
            "status": "success",
            "message": f"모델 '{model_name}' 로드 성공",
            "model": model_name,
            "backend": model_manager.active_backend,
            "loading_time": end_time - start_time
        }
    
    except Exception as e:
        logger.error(f"모델 '{model_name}' 로드 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"모델 로드 중 오류 발생: {str(e)}"
        )