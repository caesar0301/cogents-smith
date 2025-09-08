from typing import Any, Dict

from .models import AskuraState


class Memory:
    def __init__(self):
        pass

    def save(self, state: AskuraState):
        pass

    def load(self, session_id: str) -> Dict[str, Any]:
        return {}
