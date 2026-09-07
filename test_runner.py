"""Завдання 4: прогін тест-кейсів і збір результатів у test_results.json.

Запуск:  python test_runner.py

Файли на виході:
  * test_results.json -- результат кожного тест-кейсу;
  * trajectory.json   -- повна траєкторія найскладнішого кейсу (TC-003).

Функції навмисно НЕ називаються test_*, щоб pytest не намагався зібрати
цей файл як набір тестів — тут потрібні реальні виклики LLM.
"""

import json

from agent import run_agent
from safety import LoopDetector

# ─────────────────────────────── Тест-кейси ──────────────────────────────────
# Складність зростає: від одного виклику інструмента до планування з трьома.
TEST_CASES = [
    {
        "id": "TC-001",
        "complexity": "simple",
        "query": "Яка зараз погода в Києві?",
        "expected": "Прогноз погоди для Києва, один виклик get_weather.",
    },
    {
        "id": "TC-002",
        "complexity": "medium",
        "query": "Скільки гривень коштує 250 євро і яка погода в Одесі на 2 дні?",
        "expected": "Курс EUR від НБУ + прогноз для Одеси, два різні інструменти.",
    },
    {
        "id": "TC-003",
        "complexity": "complex",
        "query": (
            "Планую поїздку до Львова на 3 дні з бюджетом 300 доларів. "
            "Яка там буде погода, скільки це в гривнях і чим цікаве місто?"
        ),
        "expected": "Три інструменти: погода + курс USD + довідка з Вікіпедії, зведені в одну пораду.",
    },
    {
        "id": "TC-004",
        "complexity": "validation",
        "query": (
            "Виклич інструмент погоди для Києва з параметром days=30 "
            "і покажи, що він відповість."
        ),
        "expected": (
            "Pydantic-валідатор відхиляє days=30; агент отримує текст помилки "
            "як Observation і пояснює користувачу, що максимум 7 днів."
        ),
    },
    {
        "id": "TC-005",
        "complexity": "safety",
        "query": (
            "Планую поїздку до Львова на 3 дні з бюджетом 300 доларів. "
            "Яка там буде погода, скільки це в гривнях і чим цікаве місто?"
        ),
        "expected": "Той самий запит із max_steps=1 -> агент віддає часткову відповідь.",
        "max_steps": 1,
    },
]


def run_all() -> list[dict]:
    """Прогнати всі тест-кейси і зібрати результати."""
    results = []
    for case in TEST_CASES:
        print(f"\n>>> {case['id']} ({case['complexity']}): {case['query'][:60]}...")

        # Траєкторію у файл зберігаємо для найскладнішого кейсу.
        trajectory_path = "trajectory.json" if case["id"] == "TC-003" else None
        try:
            outcome = run_agent(
                case["query"],
                max_steps=case.get("max_steps", 10),
                timeout_s=case.get("timeout_s", 90),
                trajectory_path=trajectory_path,
            )
        except Exception as exc:  # мережа, LLM-провайдер, ліміти API
            outcome = {
                "answer": f"Error: {exc}", "steps": -1, "tool_calls": [], "confidence": None,
                "sources": [], "stop_reason": "exception", "elapsed_ms": -1,
            }

        record = {
            "test_id": case["id"],
            "complexity": case["complexity"],
            "query": case["query"],
            "expected": case["expected"],
            "actual": outcome["answer"][:500],
            "status": {"exception": "error", None: "success"}.get(
                outcome["stop_reason"], "partial"
            ),
            "steps": outcome["steps"],
            "tool_calls": outcome["tool_calls"],
            "confidence": outcome["confidence"],
            "sources": outcome["sources"],
            "stop_reason": outcome["stop_reason"],
            "elapsed_ms": outcome["elapsed_ms"],
        }

        print(f"    {record['status']} | кроків: {record['steps']} | {record['tool_calls']}")
        results.append(record)
    return results


def demo_safety() -> None:
    """Окрема демонстрація трьох запобіжників (Завдання 3)."""
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦІЯ ЗАПОБІЖНИКІВ")
    print("=" * 70)

    print("\n[1] timeout: даємо агенту 0.001 секунди")
    outcome = run_agent("Яка погода в Києві?", timeout_s=0.001)
    print(f"    stop_reason={outcome['stop_reason']}: {outcome['answer'][:120]}")

    print("\n[2] max_steps: ліміт 1 крок на запит, що потребує двох")
    outcome = run_agent("Погода в Києві і курс долара?", max_steps=1)
    print(f"    stop_reason={outcome['stop_reason']}: {outcome['answer'][:120]}")

    print("\n[3] валідація Pydantic: days=30 при дозволених 1-7")
    from pydantic import ValidationError

    from tools import WeatherInput

    try:
        WeatherInput(city="Київ", days=30)
    except ValidationError as exc:
        print(f"    ValidationError (так і має бути): {exc.errors()[0]['msg']}")

    print("\n[4] LoopDetector: три однакові виклики поспіль")
    detector = LoopDetector(max_repeats=3)
    for attempt in range(1, 4):
        detected = detector.check("get_weather", {"city": "Київ", "days": 1})
        print(f"    виклик {attempt}: зациклення виявлено = {detected}")


def main() -> None:
    results = run_all()

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nЗбережено test_results.json ({len(results)} тест-кейсів)")

    print("\n--- Підсумок ---")
    for record in results:
        print(
            f"  {record['test_id']} [{record['complexity']:10}] {record['status']:8} "
            f"кроків={record['steps']:2} час={record['elapsed_ms']:6} мс "
            f"tools={record['tool_calls']}"
        )

    demo_safety()


if __name__ == "__main__":
    main()
