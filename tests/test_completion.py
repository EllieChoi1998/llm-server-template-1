import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

# 테스트 프롬프트
TEST_PROMPT = "안녕하세요, 저는 테스트 프롬프트입니다."

# 모의 응답 데이터
MOCK_COMPLETION_RESPONSE = {
    "id": "cmpl-123456",
    "object": "text_completion",
    "created": 1677825464,
    "model": "test-model",
    "choices": [
        {
            "text": "안녕하세요, 테스트 사용자님! 어떻게 도와드릴까요?",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 20,
        "total_tokens": 35
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
        mock_instance.generate_completion.return_value = MOCK_COMPLETION_RESPONSE
        mock_instance.active_model = "test-model"
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def app_with_mock_llm(mock_llm_manager):
    """모의 LLM 매니저로 설정된 앱"""
    app.state.llm_manager = mock_llm_manager
    return app


@pytest.mark.asyncio
async def test_completion_success(app_with_mock_llm):
    """텍스트 완성 API 성공 테스트"""
    request_data = {
        "prompt": TEST_PROMPT,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/completions", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["text"] == "안녕하세요, 테스트 사용자님! 어떻게 도와드릴까요?"
    assert data["model"] == "test-model"
    assert "usage" in data
    assert data["usage"]["total_tokens"] == 35


@pytest.mark.asyncio
async def test_completion_empty_prompt():
    """텍스트 완성 API 빈 프롬프트 테스트"""
    request_data = {
        "prompt": "",
        "max_tokens": 100
    }
    
    response = client.post("/api/v1/completions", json=request_data)
    
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_completion_invalid_max_tokens():
    """텍스트 완성 API 유효하지 않은 max_tokens 값 테스트"""
    request_data = {
        "prompt": TEST_PROMPT,
        "max_tokens": -10  # 음수 값은 유효하지 않음
    }
    
    response = client.post("/api/v1/completions", json=request_data)
    
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_completion_server_error(app_with_mock_llm):
    """텍스트 완성 API 서버 오류 테스트"""
    # LLM 매니저가 예외를 발생시키도록 설정
    app_with_mock_llm.generate_completion.side_effect = Exception("테스트 오류")
    
    request_data = {
        "prompt": TEST_PROMPT,
        "max_tokens": 100
    }
    
    response = client.post("/api/v1/completions", json=request_data)
    
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data