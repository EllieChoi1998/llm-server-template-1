import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.models.request_models import ChatMessage

client = TestClient(app)

# 테스트 채팅 메시지
TEST_MESSAGES = [
    ChatMessage(role="system", content="당신은 유용한 AI 어시스턴트입니다."),
    ChatMessage(role="user", content="안녕하세요, 저는 테스트 사용자입니다.")
]

# 모의 응답 데이터
MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl-123456",
    "object": "chat.completion",
    "created": 1677825464,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "안녕하세요, 테스트 사용자님! 어떻게 도와드릴까요?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 20,
        "total_tokens": 45
    },
    "system_info": {
        "processing_time": 0.5
    }
}


@pytest.fixture
def mock_llm_manager():
    """LLM 매니저 모의 객체 생성"""
    with patch('app.main.LLMManager') as mock:
        mock_instance = MagicMock()
        mock_instance.generate_chat_completion.return_value = MOCK_CHAT_RESPONSE
        mock_instance.active_model = "test-model"
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def app_with_mock_llm(mock_llm_manager):
    """모의 LLM 매니저로 설정된 앱"""
    app.state.llm_manager = mock_llm_manager
    return app


@pytest.mark.asyncio
async def test_chat_completion_success(app_with_mock_llm):
    """채팅 완성 API 성공 테스트"""
    request_data = {
        "messages": [msg.dict() for msg in TEST_MESSAGES],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/chat/completions", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "안녕하세요, 테스트 사용자님! 어떻게 도와드릴까요?"
    assert data["model"] == "test-model"
    assert "usage" in data
    assert data["usage"]["total_tokens"] == 45


@pytest.mark.asyncio
async def test_chat_completion_invalid_request():
    """채팅 완성 API 유효하지 않은 요청 테스트"""
    # 빈 메시지 목록으로 요청
    request_data = {
        "messages": [],
        "max_tokens": 100
    }
    
    response = client.post("/api/v1/chat/completions", json=request_data)
    
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_chat_completion_invalid_temperature():
    """채팅 완성 API 유효하지 않은 온도 값 테스트"""
    request_data = {
        "messages": [msg.dict() for msg in TEST_MESSAGES],
        "temperature": 3.0  # 유효 범위를 벗어남 (0.0-2.0)
    }
    
    response = client.post("/api/v1/chat/completions", json=request_data)
    
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_chat_completion_server_error(app_with_mock_llm):
    """채팅 완성 API 서버 오류 테스트"""
    # LLM 매니저가 예외를 발생시키도록 설정
    app_with_mock_llm.generate_chat_completion.side_effect = Exception("테스트 오류")
    
    request_data = {
        "messages": [msg.dict() for msg in TEST_MESSAGES],
        "max_tokens": 100
    }
    
    response = client.post("/api/v1/chat/completions", json=request_data)
    
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data