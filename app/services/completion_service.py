from typing import Dict, Any, List, Optional
import logging

from app.utils.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

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
        logger.info(f"텍스트 완성 생성 시작, 모델: {model or '기본'}")
        
        # 요청 검증
        if not prompt or not prompt.strip():
            raise InvalidRequestError("프롬프트는 비어 있을 수 없습니다")
        
        try:
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
            
            logger.info(f"텍스트 완성 생성 완료, 토큰 사용: {response['usage']['total_tokens']}")
            return response
            
        except Exception as e:
            logger.error(f"텍스트 완성 생성 중 오류: {str(e)}")
            raise