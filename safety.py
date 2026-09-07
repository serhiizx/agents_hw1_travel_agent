"""Завдання 3: захисні механізми агента.

Три незалежні запобіжники:
  * MAX_STEPS      -- скільки разів агент може пройти цикл LLM -> tools -> LLM;
  * TIMEOUT_SECONDS -- загальний бюджет часу на один запит;
  * LoopDetector   -- ловить ситуацію, коли LLM вперто кличе той самий
                      інструмент з тими самими аргументами і чекає іншого
                      результату.
"""

import hashlib

# Ліміт ітерацій ReAct-циклу. 10 вистачає навіть на складний запит із 3 tools.
MAX_STEPS = 10

# Загальний тайм-аут на один запит користувача, секунд.
TIMEOUT_SECONDS = 90


class LoopDetector:
    """Детекція зациклення: N однакових tool calls поспіль."""

    def __init__(self, max_repeats: int = 3) -> None:
        self.recent_calls: list[str] = []
        self.max_repeats = max_repeats

    def reset(self) -> None:
        """Забути історію викликів перед новим запитом."""
        self.recent_calls.clear()

    def check(self, tool_name: str, args: dict) -> bool:
        """Зареєструвати виклик і сказати, чи це вже зациклення.

        Returns:
            True, якщо останні max_repeats викликів були абсолютно однакові.
        """
        call_hash = hashlib.md5(
            f"{tool_name}:{sorted(args.items())}".encode()
        ).hexdigest()
        self.recent_calls.append(call_hash)

        if len(self.recent_calls) >= self.max_repeats:
            last_n = self.recent_calls[-self.max_repeats:]
            if len(set(last_n)) == 1:
                return True
        return False
