class OllamaConnectionError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class OllamaModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class OllamaResourceNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

