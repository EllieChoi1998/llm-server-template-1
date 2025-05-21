class ModelNotLoadedError(Exception):
    """모델이 로드되지 않았을 때 발생하는 예외"""
    def __init__(self, message="모델이 로드되지 않았습니다"):
        self.message = message
        super().__init__(self.message)


class InferenceError(Exception):
    """모델 추론 중 오류가 발생했을 때 발생하는 예외"""
    def __init__(self, message="모델 추론 중 오류가 발생했습니다"):
        self.message = message
        super().__init__(self.message)


class InvalidRequestError(Exception):
    """유효하지 않은 요청에 대한 예외"""
    def __init__(self, message="유효하지 않은 요청입니다"):
        self.message = message
        super().__init__(self.message)


class FileNotFoundError(Exception):
    """파일을 찾을 수 없을 때 발생하는 예외"""
    def __init__(self, message="파일을 찾을 수 없습니다"):
        self.message = message
        super().__init__(self.message)


class AuthenticationError(Exception):
    """인증 오류에 대한 예외"""
    def __init__(self, message="인증에 실패했습니다"):
        self.message = message
        super().__init__(self.message)


class RateLimitError(Exception):
    """요청 제한에 도달했을 때 발생하는 예외"""
    def __init__(self, message="요청 제한에 도달했습니다"):
        self.message = message
        super().__init__(self.message)


class ServiceUnavailableError(Exception):
    """서비스를 사용할 수 없을 때 발생하는 예외"""
    def __init__(self, message="서비스를 사용할 수 없습니다"):
        self.message = message
        super().__init__(self.message)