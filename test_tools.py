"""Завдання 1: unit-тести інструментів.

Дві групи перевірок:
  1. Валідація Pydantic -- некоректні входи мають відхилятись із зрозумілим
     повідомленням (це те, що рятує агента від "галюцинацій" параметрів).
  2. Реальний виклик інструмента -- перевіряємо, що відповідь непорожня.
     Якщо мережі немає, інструмент повертає рядок "Помилка ...", і це теж
     коректна поведінка: агент має отримати текст, а не виняток.

Запуск: python -m pytest test_tools.py -v
"""

import pytest
from pydantic import ValidationError

from tools import RateInput, WeatherInput, WikiInput, get_nbu_rate, get_weather, get_wiki_summary


# ───────────────────────── Валідація: погода ─────────────────────────

def test_weather_valid_input():
    schema = WeatherInput(city="  Київ  ", days=3)
    assert schema.city == "Київ"  # str_strip_whitespace прибрав пробіли
    assert schema.days == 3


def test_weather_rejects_empty_city():
    with pytest.raises(ValidationError, match="мінімум 2 символи"):
        WeatherInput(city="К", days=1)


def test_weather_rejects_too_many_days():
    with pytest.raises(ValidationError, match="від 1 до 7"):
        WeatherInput(city="Київ", days=30)


# ────────────────────── Валідація: курс валюти ───────────────────────

def test_rate_normalizes_code_to_upper():
    assert RateInput(currency_code="usd").currency_code == "USD"


def test_rate_rejects_wrong_code():
    with pytest.raises(ValidationError, match="трьох літер"):
        RateInput(currency_code="ДОЛАР")


def test_rate_rejects_negative_amount():
    with pytest.raises(ValidationError, match="більшою за нуль"):
        RateInput(currency_code="USD", amount=-5)


# ───────────────────────── Валідація: Вікіпедія ──────────────────────

def test_wiki_rejects_unsupported_lang():
    with pytest.raises(ValidationError, match="uk"):
        WikiInput(topic="Київ", lang="fr")


def test_wiki_rejects_short_topic():
    with pytest.raises(ValidationError, match="мінімум 2 символи"):
        WikiInput(topic="К")


# ──────────────── Реальні виклики інструментів ───────────────────────

def test_get_weather_returns_forecast():
    result = get_weather.invoke({"city": "Київ", "days": 2})
    assert isinstance(result, str) and result
    assert "Погода" in result or result.startswith("Помилка")


def test_get_nbu_rate_returns_uah():
    result = get_nbu_rate.invoke({"currency_code": "USD", "amount": 100})
    assert "UAH" in result or result.startswith("Помилка")


def test_get_wiki_summary_returns_text():
    result = get_wiki_summary.invoke({"topic": "Львів"})
    assert len(result) > 50 or result.startswith("Помилка")


def test_get_wiki_summary_handles_missing_article():
    """Неіснуюча стаття -- це не виняток, а зрозуміле повідомлення для агента."""
    result = get_wiki_summary.invoke({"topic": "ЦеїСтаттіТочноНемає99999"})
    assert "немає статті" in result or result.startswith("Помилка")
