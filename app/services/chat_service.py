from typing import List, Dict, Any, Optional
import logging
import time

from app.models.request_models import ChatMessage
from app.utils.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

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
        logger.info(f"채팅 완성 생성 시작, 모델: {model or '기본'}")
        
        # 요청 검증
        if not messages:
            raise InvalidRequestError("메시지 목록은 비어 있을 수 없습니다")
        
        # 메시지 형식 변환
        formatted_messages = [msg.dict() for msg in messages]
        
        try:
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
            
            logger.info(f"채팅 완성 생성 완료, 토큰 사용: {response['usage']['total_tokens']}")
            return response
            
        except Exception as e:
            logger.error(f"채팅 완성 생성 중 오류: {str(e)}")
            raise