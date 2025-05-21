from typing import List, Tuple, Optional
import logging
import time

from app.utils.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

class EmbeddingService:
    """텍스트 임베딩 서비스"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> Tuple[List[float], int]:
        """텍스트의 임베딩을 생성합니다.
        
        반환값:
            Tuple[List[float], int]: 임베딩 벡터와 토큰 수
        """
        logger.info(f"임베딩 생성 시작, 모델: {model or '기본'}")
        
        # 요청 검증
        if not text or not text.strip():
            raise InvalidRequestError("임베딩할 텍스트는 비어 있을 수 없습니다")
        
        try:
            start_time = time.time()
            
            # 모델 매니저를 통해 임베딩 생성
            embeddings = await self.model_manager.get_embeddings(text, model_name=model)
            
            # 토큰 수 추정 (실제로는 모델별로 다를 수 있음)
            # 간단한 휴리스틱: 평균적으로 영어 텍스트에서 4글자당 약 1토큰
            token_count = max(1, len(text) // 4)
            
            end_time = time.time()
            logger.info(f"임베딩 생성 완료: 벡터 크기 {len(embeddings)}, 시간 {end_time - start_time:.3f}초")
            
            return embeddings, token_count
            
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류: {str(e)}")
            logger.info(f"임베딩 생성 완료: 벡터 크기 {len(embeddings)}, 시간 {end_time - start_time:.3f}초")
            
            return embeddings, token_count
            
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류: {str(e)}")
            raise