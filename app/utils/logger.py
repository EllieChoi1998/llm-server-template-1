import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import time
from datetime import datetime

from app.config import settings

# 로그 포맷 설정
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 로그 레벨 매핑
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


def setup_logger(name: str = None):
    """로거 설정

    Args:
        name (str, optional): 로거 이름. 기본값은 None.

    Returns:
        logging.Logger: 설정된 로거
    """
    # 로거 이름이 없으면 기본 이름 사용
    if name is None:
        name = "app"

    # 로그 레벨 설정
    log_level = LOG_LEVELS.get(settings.LOG_LEVEL, logging.INFO)
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 이미 핸들러가 있으면 중복 방지를 위해 건너뜀
    if logger.handlers:
        return logger
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 설정 (선택 사항)
    if settings.LOG_FILE:
        log_file = Path(settings.LOG_FILE)
        
        # 로그 디렉토리가 없으면 생성
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 일별 로그 파일 설정
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=7,  # 7일치 로그 유지
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 대용량 파일 로테이션 설정
        size_handler = RotatingFileHandler(
            filename=log_file.with_suffix(".size.log"),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        size_handler.setLevel(log_level)
        size_handler.setFormatter(file_formatter)
        logger.addHandler(size_handler)
    
    # 로거가 상위 로거로 전파하지 않도록 설정
    logger.propagate = False
    
    return logger


def get_request_id():
    """고유한 요청 ID 생성"""
    timestamp = int(time.time() * 1000)
    return f"req-{timestamp}"


class RequestLogger:
    """요청별 로깅 클래스"""
    
    def __init__(self, request_id=None):
        self.logger = setup_logger("app.request")
        self.request_id = request_id or get_request_id()
    
    def info(self, message, **kwargs):
        """정보 레벨 로그"""
        self._log(logging.INFO, message, **kwargs)
    
    def debug(self, message, **kwargs):
        """디버그 레벨 로그"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def warning(self, message, **kwargs):
        """경고 레벨 로그"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message, **kwargs):
        """에러 레벨 로그"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message, **kwargs):
        """치명적 에러 레벨 로그"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level, message, **kwargs):
        """내부 로깅 메서드"""
        extra = {"request_id": self.request_id, **kwargs}
        self.logger.log(level, f"[{self.request_id}] {message}", extra=extra)