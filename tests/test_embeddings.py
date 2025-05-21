import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

# 테스트 입력 텍스트
TEST_TEXT = "이것은 임베딩 테스트를 위한 텍스트입니다."
TEST_TEXTS = ["첫 번째 텍스트입니다.", "두 번째 텍스트입니다."]

# 모의 임베딩 벡터
MOCK_EMBEDDING = [0.01, 0.02, 0.03, 0.04, 0.05]


@pytest.fixture
def mock_llm_manager():
    """LLM 매니저 모의 객체 생성"""
    with patch('app.main.LLMManager') as mock:
        mock_instance = MagicMock()
        mock_instance.get_embeddings.return_value = MOCK_EMBEDDING
        mock_instance.active_model = "test-model"
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def app_with_mock_llm(mock_llm_manager):
    """모의 LLM 매니저로 설정된 앱"""
    app.state.llm_manager = mock_llm_manager
    return app


@pytest.mark.asyncio
async def test_single_embedding_success(app_with_mock_llm):
    """단일 텍스트 임베딩 API 성공 테스트"""
    request_data = {
        "input": TEST_TEXT
    }
    
    response = client.post("/api/v1/embeddings", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["embedding"] == MOCK_EMBEDDING
    assert data["model"] == "test-model"
    assert "usage" in data


@pytest.mark.asyncio
async def test_multiple_embeddings_success(app_with_mock_llm):
    """다중 텍스트 임베딩 API 성공 테스트"""
    request_data = {
        "input": TEST_TEXTS
    }
    
    response = client.post("/api/v1/embeddings", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == len(TEST_TEXTS)
    assert data["data"][0]["embedding"] == MOCK_EMBEDDING
    assert data["data"][1]["embedding"] == MOCK_EMBEDDING
    assert data["model"] == "test-model"
    assert "usage" in data


@pytest.mark.asyncio
async def test_embedding_empty_input():
    """임베딩 API 빈 입력 테스트"""
    request_data = {
        "input": ""
    }
    
    response = client.post("/api/v1/embeddings", json=request_data)
    
    assert response.status_code == 422  # Unprocessable Entity
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_embedding_empty_list():
    """임베딩 API 빈 목록 입력 테스트"""
    request_data = {
        "input": []
    }
    
    response = client.post("/api/v1/embeddings", json=request_data)
    
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_embedding_server_error(app_with_mock_llm):
    """임베딩 API 서버 오류 테스트"""
    # LLM 매니저가 예외를 발생시키도록 설정
    app_with_mock_llm.get_embeddings.side_effect = Exception("테스트 오류")
    
    request_data = {
        "input": TEST_TEXT
    }
    
    response = client.post("/api/v1/embeddings", json=request_data)
    
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data