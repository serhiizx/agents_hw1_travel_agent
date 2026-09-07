"""Завдання 3: логування траєкторії виконання агента у JSON.

TrajectoryLogger накопичує кроки під час роботи графа і зберігає їх у файл.
Це потрібно для post-mortem аналізу: якщо агент дав дивну відповідь, за
trajectory.json видно, який саме крок пішов не так.
"""

import json
import time
from datetime import datetime, timezone

# Максимальна довжина текстових полів у лозі, щоб файл не розпух.
MAX_FIELD_LEN = 500


class TrajectoryLogger:
    """Накопичує кроки виконання агента і зберігає їх у JSON-файл."""

    def __init__(self) -> None:
        self.steps: list[dict] = []
        self.start_time = time.monotonic()

    def log_step(
        self,
        step_num: int,
        node: str,
        input_data: str,
        output_data: str,
        tool_calls: list | None = None,
    ) -> None:
        """Записати один крок виконання (виклик одного вузла графа)."""
        self.steps.append(
            {
                "step": step_num,
                "node": node,
                "input": str(input_data)[:MAX_FIELD_LEN],
                "output": str(output_data)[:MAX_FIELD_LEN],
                "tool_calls": tool_calls or [],
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "elapsed_ms": int((time.monotonic() - self.start_time) * 1000),
            }
        )

    def as_dict(self, query: str = "", stop_reason: str | None = None) -> dict:
        """Повернути траєкторію як словник (для вкладення в інші JSON-и)."""
        return {
            "query": query,
            "stop_reason": stop_reason,
            "total_steps": len(self.steps),
            "total_time_ms": int((time.monotonic() - self.start_time) * 1000),
            "trajectory": self.steps,
        }

    def save(self, filepath: str, query: str = "", stop_reason: str | None = None) -> None:
        """Зберегти траєкторію у JSON-файл (ensure_ascii=False — щоб була кирилиця)."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(query, stop_reason), f, ensure_ascii=False, indent=2)
        print(f"Траєкторію збережено: {filepath} ({len(self.steps)} кроків)")
