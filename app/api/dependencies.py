from fastapi import Header, HTTPException, Depends, Request
from typing import Optional
import logging

from app.config import settings
from app.utils.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

async def get_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None)
) -> Optional[str]:
    """API 키 검증
    
    API 키가 필요한 경우 헤더에서 키를 가져와 검증합니다.
    """
    # API 키가 필요하지 않은 경우 무시
    if not settings.API_KEY_REQUIRED:
        return None
    
    if not x_api_key:
        logger.warning("API 키가 필요하지만 제공되지 않았습니다")
        raise HTTPException(
            status_code=401,
            detail="API 키가 필요합니다. 'X-API-Key' 헤더를 통해 제공해주세요."
        )
    
    # API 키 검증
    if x_api_key != settings.API_KEY:
        logger.warning("잘못된 API 키가 제공되었습니다")
        raise HTTPException(
            status_code=401,
            detail="API 키가 유효하지 않습니다."
        )
    
    return x_api_key