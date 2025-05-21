from fastapi import APIRouter, Request, Depends, HTTPException
from typing import Dict, Any, Optional, List
import logging
import time
import asyncio

from app.models.request_models import EmbeddingRequest
from app.services.embedding_service import EmbeddingService
from app.api.dependencies import get_api_key

router = APIRouter()
logger = logging.getLogger(__name__)

# 세마포어를 사용하여 동시 요청 제한
semaphore = asyncio.Semaphore(10)  # 임베딩은 좀 더 많은 동시 요청 허용

@router.post("/embeddings", response_model=Dict[str, Any])
async def create_embeddings(
    request: Request,
    embedding_request: EmbeddingRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """텍스트 임베딩 생성 엔드포인트"""
    start_time = time.time()
    
    # 모델 매니저 가져오기
    model_manager = request.app.state.model_manager
    
    # 임베딩 서비스 생성
    embedding_service = EmbeddingService(model_manager)
    
    try:
        # 세마포어 획득 (동시 요청 제한)
        async with semaphore:
            # 텍스트 또는 텍스트 목록 처리
            if isinstance(embedding_request.input, str):
                logger.info(f"단일 텍스트 임베딩 요청 받음: 길이 {len(embedding_request.input)}")
                # 단일 텍스트 임베딩 생성
                embeddings, token_count = await embedding_service.generate_embedding(
                    text=embedding_request.input,
                    model=embedding_request.model
                )
                data = [{"embedding": embeddings, "index": 0, "object": "embedding"}]
            else:
                logger.info(f"텍스트 목록 임베딩 요청 받음: 항목 수 {len(embedding_request.input)}")
                # 텍스트 목록 임베딩 생성
                results = []
                token_count = 0
                
                # 모든 텍스트에 대해 임베딩 생성
                for i, text in enumerate(embedding_request.input):
                    embedding, tokens = await embedding_service.generate_embedding(
                        text=text,
                        model=embedding_request.model
                    )
                    results.append({
                        "embedding": embedding,
                        "index": i,
                        "object": "embedding"
                    })
                    token_count += tokens
                
                data = results
            
            # 응답 시간 기록
            end_time = time.time()
            process_time = end_time - start_time
            
            # 응답 형식화
            response = {
                "object": "list",
                "data": data,
                "model": embedding_request.model or model_manager.active_model,
                "usage": {
                    "prompt_tokens": token_count,
                    "total_tokens": token_count
                }
            }
            
            logger.info(f"임베딩 생성 완료: 처리 시간 {process_time:.3f}초, 토큰 수 {token_count}")
            return response
    
    except Exception as e:
        logger.error(f"임베딩 생성 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))