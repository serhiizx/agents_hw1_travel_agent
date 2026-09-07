"""Завдання 1: інструменти агента-помічника мандрівника.

Кожен інструмент має:
  * Pydantic v2 схему параметрів з Field(description=...) -- саме опис читає
    LLM, коли вирішує, який інструмент викликати і що в нього передати;
  * field_validator -- захист від "галюцинацій" параметрів (порожнє місто,
    30 днів прогнозу, неіснуючий код валюти);
  * докладний docstring з прикладом -- це друга половина того, що бачить LLM.

Усі інструменти працюють з безкоштовними публічними API без ключів.
Мережеві помилки НЕ кидаються назовні, а повертаються текстом: агент має
прочитати їх як Observation і вирішити, що робити далі, а не впасти.
"""

from urllib.parse import quote

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

HTTP_TIMEOUT = 10.0

# Вікіпедія віддає 403 на запити без User-Agent, тому представляємось.
# Значення заголовка має бути ASCII, тому латиницею.
HTTP_HEADERS = {"User-Agent": "hw1-react-agent/1.0 (educational project)"}


def _get_json(url: str, params: dict | None = None):
    """Один GET-запит із коротким тайм-аутом. Кидає httpx.HTTPError при збої."""
    response = httpx.get(
        url,
        params=params,
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.json()


# ─────────────────────────── Інструмент 1: погода ───────────────────────────

class WeatherInput(BaseModel):
    """Параметри запиту прогнозу погоди."""

    model_config = ConfigDict(str_strip_whitespace=True)

    city: str = Field(description='Назва міста українською або англійською, наприклад "Київ" або "Rome".')
    days: int = Field(default=1, description="Кількість днів прогнозу, від 1 до 7.")

    @field_validator("city")
    @classmethod
    def city_not_empty(cls, v: str) -> str:
        if len(v) < 2:
            raise ValueError("Назва міста повинна містити мінімум 2 символи")
        return v

    @field_validator("days")
    @classmethod
    def days_in_range(cls, v: int) -> int:
        if not 1 <= v <= 7:
            raise ValueError("Кількість днів прогнозу має бути від 1 до 7")
        return v


@tool(args_schema=WeatherInput)
def get_weather(city: str, days: int = 1) -> str:
    """Отримати прогноз погоди для міста на 1-7 днів.

    Використовуй цей інструмент, коли користувач питає про погоду, температуру,
    дощ або що вдягнути в конкретному місті.

    Приклад: get_weather(city="Львів", days=3) -> прогноз на три доби.

    Args:
        city: Назва міста.
        days: Кількість днів прогнозу (1-7).

    Returns:
        Текстовий прогноз: макс/мін температура та опади по днях.
    """
    try:
        geo = _get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": city, "count": 1, "language": "uk"},
        )
        places = geo.get("results") or []
        if not places:
            return f"Місто «{city}» не знайдено в геокодері. Перевір назву."

        place = places[0]
        forecast = _get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": days,
                "timezone": "auto",
            },
        )
    except httpx.HTTPError as exc:
        return f"Помилка запиту погоди: {exc}"

    daily = forecast["daily"]
    rows = [
        f"{date}: від {tmin}°C до {tmax}°C, опади {rain} мм"
        for date, tmax, tmin, rain in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_sum"],
        )
    ]
    header = f"Погода: {place['name']}, {place.get('country', '')}".strip().rstrip(",")
    return header + "\n" + "\n".join(rows)


# ──────────────────────── Інструмент 2: курс валюти НБУ ──────────────────────

class RateInput(BaseModel):
    """Параметри запиту офіційного курсу валюти НБУ."""

    model_config = ConfigDict(str_strip_whitespace=True)

    currency_code: str = Field(description='Код валюти ISO 4217 із трьох літер, наприклад "USD", "EUR", "PLN".')
    amount: float = Field(default=1.0, description="Скільки одиниць валюти перерахувати в гривні. Має бути більше 0.")

    @field_validator("currency_code")
    @classmethod
    def code_is_three_letters(cls, v: str) -> str:
        if len(v) != 3 or not v.isalpha():
            raise ValueError("Код валюти має складатися рівно з трьох літер, наприклад USD")
        return v.upper()

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Сума має бути більшою за нуль")
        return v


@tool(args_schema=RateInput)
def get_nbu_rate(currency_code: str, amount: float = 1.0) -> str:
    """Отримати офіційний курс валюти до гривні за даними Національного банку України.

    Використовуй цей інструмент, коли користувач питає курс валюти, скільки
    коштує долар/євро, або скільки гривень треба на певну суму в валюті.

    Приклад: get_nbu_rate(currency_code="EUR", amount=250) -> скільки це гривень.

    Args:
        currency_code: Код валюти з трьох літер (USD, EUR, PLN, GBP...).
        amount: Сума у цій валюті для перерахунку в гривні.

    Returns:
        Курс НБУ, дата курсу та результат перерахунку.
    """
    try:
        data = _get_json(
            "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange",
            {"valcode": currency_code, "json": ""},
        )
    except httpx.HTTPError as exc:
        return f"Помилка запиту курсу: {exc}"

    if not data:
        return f"НБУ не знає валюти «{currency_code}». Перевір код."

    row = data[0]
    total = row["rate"] * amount
    return (
        f"{row['txt']} ({row['cc']}): 1 {row['cc']} = {row['rate']:.4f} UAH "
        f"на {row['exchangedate']}. {amount:g} {row['cc']} = {total:.2f} UAH."
    )


# ──────────────────── Інструмент 3: коротка довідка з Вікіпедії ───────────────

class WikiInput(BaseModel):
    """Параметри запиту короткої довідки з Вікіпедії."""

    model_config = ConfigDict(str_strip_whitespace=True)

    topic: str = Field(description='Тема або назва статті, наприклад "Львів" або "Ейфелева вежа".')
    lang: str = Field(default="uk", description='Мова Вікіпедії: "uk" або "en".')

    @field_validator("topic")
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        if len(v) < 2:
            raise ValueError("Тема має містити мінімум 2 символи")
        return v

    @field_validator("lang")
    @classmethod
    def lang_supported(cls, v: str) -> str:
        v = v.lower()
        if v not in {"uk", "en"}:
            raise ValueError('Підтримуються лише мови "uk" та "en"')
        return v


@tool(args_schema=WikiInput)
def get_wiki_summary(topic: str, lang: str = "uk") -> str:
    """Отримати коротку довідку з Вікіпедії про місто, країну або пам'ятку.

    Використовуй цей інструмент, коли треба фактична довідка: що це за місто,
    чим відоме, що подивитись. Не використовуй для погоди чи курсів валют.

    Приклад: get_wiki_summary(topic="Одеса") -> два абзаци про місто.

    Args:
        topic: Тема статті.
        lang: Мова Вікіпедії ("uk" або "en").

    Returns:
        Перший абзац статті з посиланням на неї.
    """
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(topic)}"
    try:
        data = _get_json(url)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return f"У Вікіпедії ({lang}) немає статті «{topic}»."
        return f"Помилка запиту до Вікіпедії: {exc}"
    except httpx.HTTPError as exc:
        return f"Помилка запиту до Вікіпедії: {exc}"

    extract = data.get("extract")
    if not extract:
        return f"Стаття «{topic}» знайдена, але без тексту."
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    return f"{data.get('title', topic)}: {extract}\nДжерело: {page_url}"


# Список інструментів, який отримує LLM через bind_tools.
TOOLS = [get_weather, get_nbu_rate, get_wiki_summary]
