from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import logging
import time
import asyncio

from app.models.request_models import CompletionRequest
from app.services.completion_service import CompletionService
from app.api.dependencies import get_api_key

router = APIRouter()
logger = logging.getLogger(__name__)

# 세마포어를 사용하여 동시 요청 제한
semaphore = asyncio.Semaphore(5)  # 최대 5개의 동시 요청 허용

@router.post("/completions", response_model=Dict[str, Any])
async def create_completion(
    request: Request,
    completion_request: CompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    """텍스트 완성 생성 엔드포인트"""
    logger.info(f"텍스트 완성 요청 받음: 프롬프트 길이 {len(completion_request.prompt)}")
    
    start_time = time.time()
    
    # 모델 매니저 가져오기
    model_manager = request.app.state.model_manager
    
    # 완성 서비스 생성
    completion_service = CompletionService(model_manager)
    
    try:
        # 세마포어 획득 (동시 요청 제한)
        async with semaphore:
            # 텍스트 완성 생성
            response = await completion_service.generate_completion(
                prompt=completion_request.prompt,
                model=completion_request.model,
                max_tokens=completion_request.max_tokens,
                temperature=completion_request.temperature,
                top_p=completion_request.top_p,
                top_k=completion_request.top_k,
                stop=completion_request.stop
            )
            
            # 응답 시간 기록
            end_time = time.time()
            process_time = end_time - start_time
            logger.info(f"텍스트 완성 생성 완료: 처리 시간 {process_time:.3f}초")
            
            # 응답 로깅을 백그라운드 작업으로 추가
            background_tasks.add_task(
                log_completion,
                prompt=completion_request.prompt,
                response_text=response["choices"][0]["text"] if response["choices"] else "",
                model=completion_request.model,
                process_time=process_time
            )
            
            return response
    
    except Exception as e:
        logger.error(f"텍스트 완성 생성 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions/stream", response_model=Dict[str, Any])
async def create_completion_stream(
    request: Request,
    completion_request: CompletionRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """텍스트 완성 스트림 생성 엔드포인트 (SSE)"""
    logger.info(f"텍스트 완성 스트림 요청 받음: 프롬프트 길이 {len(completion_request.prompt)}")
    
    # 현재 이 API는 아직 개발 중입니다
    raise HTTPException(
        status_code=501, 
        detail="스트리밍 API는 개발 중입니다. 현재 지원되지 않습니다."
    )


async def log_completion(
    prompt: str,
    response_text: str,
    model: str,
    process_time: float
):
    """텍스트 완성 로깅을 위한 백그라운드 작업"""
    # 프로덕션 환경에서는 구조화된 로깅 시스템이나 DB에 저장할 수 있음
    logger.debug(f"텍스트 완성 로그: 모델={model}, 시간={process_time:.3f}초")
    
    # 여기에 로깅 또는 분석 로직 추가