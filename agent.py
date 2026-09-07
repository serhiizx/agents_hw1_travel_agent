r"""Завдання 2 + 3: ReAct-агент на LangGraph.

Граф рівно такий, як у прикладі з ДЗ — два вузли й один conditional edge:

    START -> agent --(є tool_calls?)--> tools -> agent
                   \--(немає)--------> END

  * agent -- LLM вирішує: викликати інструмент чи вже відповідати. Тут же
             спрацьовують запобіжники max_steps / timeout / зациклення.
  * tools -- вбудований ToolNode виконує викликані інструменти.

Structured output робиться після завершення графа, у run_agent: остання
репліка агента переганяється у Pydantic-модель FinalAnswer.

Логування траєкторії теж живе у run_agent: app.stream(stream_mode="updates")
віддає оновлення по одному на кожен відпрацьований вузол, тому вузли
лишаються чистими функціями без домішки логування.
"""

import os
import time
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from logger import TrajectoryLogger
from safety import MAX_STEPS, TIMEOUT_SECONDS, LoopDetector
from tools import TOOLS

load_dotenv()

SYSTEM_PROMPT = """Ти — помічник мандрівника. Відповідай українською мовою.

У тебе є три інструменти:
  * get_weather      — прогноз погоди в місті на 1-7 днів;
  * get_nbu_rate     — офіційний курс валюти до гривні від НБУ;
  * get_wiki_summary — коротка довідка з Вікіпедії про місто чи пам'ятку.

Правила:
  * Не вигадуй числа. Погоду, курси й факти бери ТІЛЬКИ з інструментів.
  * Якщо питання охоплює кілька тем — виклич кілька інструментів.
  * Якщо інструмент повернув повідомлення про помилку — скажи про це чесно,
    не повторюй той самий виклик з тими самими аргументами.
  * Коли даних достатньо — дай коротку зв'язну відповідь без зайвої води.
"""


class FinalAnswer(BaseModel):
    """Структурована фінальна відповідь агента (Завдання 2, structured output)."""

    answer: str = Field(description="Готова відповідь користувачу українською мовою.")
    confidence: float = Field(description="Наскільки агент упевнений у відповіді, від 0.0 до 1.0.")
    sources: list[str] = Field(
        default_factory=list,
        description="Назви використаних інструментів або посилання на джерела.",
    )


class AgentState(TypedDict):
    """Стан, який передається між вузлами графа."""

    messages: Annotated[list, add_messages]
    step_count: int          # скільки разів відпрацював вузол agent
    max_steps: int           # ліміт кроків саме цього запуску
    deadline: float          # time.monotonic(), після якого треба зупинитись
    stop_reason: str | None  # None | "max_steps" | "timeout" | "loop"


# ── LLM та інструменти ───────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
    api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
    temperature=0.1,
)
llm_with_tools = llm.bind_tools(TOOLS)

# Детектор зациклення один на модуль — run_agent скидає його перед запуском.
# ponytail: не потоко-безпечно; для паралельних запитів довелося б тримати
# детектор у стані графа або збирати граф на кожен запит.
detector = LoopDetector(max_repeats=3)


# ── Вузол agent ──────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """LLM приймає рішення: викликати tool чи відповісти. Плюс три запобіжники."""
    step = state["step_count"]

    if step >= state["max_steps"]:
        return {
            "messages": [AIMessage(content=(
                f"Досягнуто ліміт кроків ({state['max_steps']}). Це часткова відповідь: "
                f"агент не встиг зібрати всі дані."
            ))],
            "stop_reason": "max_steps",
        }

    if time.monotonic() > state["deadline"]:
        return {
            "messages": [AIMessage(content="Вичерпано час на обробку запиту. Це часткова відповідь.")],
            "stop_reason": "timeout",
        }

    response = llm_with_tools.invoke(state["messages"])

    for call in getattr(response, "tool_calls", None) or []:
        if detector.check(call["name"], call["args"]):
            return {
                "messages": [AIMessage(content=(
                    f"Виявлено зациклення: інструмент {call['name']} викликано "
                    f"з тими самими аргументами тричі поспіль. Зупиняюсь."
                ))],
                "step_count": step + 1,
                "stop_reason": "loop",
            }

    return {"messages": [response], "step_count": step + 1}


# ── Router: серце ReAct-циклу ────────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Є tool_calls -> виконуємо інструменти. Немає -> завершуємо граф."""
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "__end__"


# ── Граф ─────────────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()


def _describe(messages: list) -> tuple[str, list[str]]:
    """Текст і список інструментів для одного запису траєкторії."""
    text = "\n".join(str(m.content) for m in messages if str(m.content))
    names = [c["name"] for m in messages for c in (getattr(m, "tool_calls", None) or [])]
    names += [m.name for m in messages if getattr(m, "name", None)]
    return text or "(тільки виклики інструментів)", names


# Префікси, з яких починається Observation з помилкою: або наш власний текст
# із інструмента, або повідомлення ToolNode про відхилену Pydantic-валідацію.
ERROR_PREFIXES = ("Помилка", "Error invoking tool")


def build_final_answer(
    text: str, used_tools: list[str], observations: list[str], stop_reason: str | None
) -> FinalAnswer:
    """Зібрати структуровану відповідь із фактів прогону, без другого виклику LLM.

    Раніше тут був окремий llm.with_structured_output(...). Від нього відмовились
    із двох причин. По-перше, модель час від часу відповідала на службовий промт
    замість того, щоб переформатувати готову відповідь агента. По-друге, її
    самооцінка confidence на всіх тест-кейсах дорівнювала рівно 1.0, тобто не
    несла жодної інформації.

    Тепер confidence — проста евристика за тим, що реально сталося:
      0.0 — спрацював запобіжник, відповідь свідомо неповна;
      0.3 — жоден інструмент не викликано, відповідь із власних знань моделі;
      0.5 — хоча б один інструмент повернув помилку;
      0.9 — усі викликані інструменти відпрацювали чисто.
    """
    if stop_reason:
        gathered = "\n".join(observations) or "жодних даних зібрати не встигли"
        return FinalAnswer(
            answer=f"{text}\n\nЗібрані дані:\n{gathered}", confidence=0.0, sources=used_tools
        )

    if not used_tools:
        confidence = 0.3
    elif any(o.startswith(ERROR_PREFIXES) for o in observations):
        confidence = 0.5
    else:
        confidence = 0.9
    return FinalAnswer(answer=text, confidence=confidence, sources=used_tools)


def run_agent(
    query: str,
    max_steps: int = MAX_STEPS,
    timeout_s: float = TIMEOUT_SECONDS,
    trajectory_path: str | None = None,
) -> dict:
    """Прогнати один запит через агента і повернути результат із траєкторією.

    max_steps і timeout_s задаються на кожен запуск окремо — так test_runner
    може форсувати спрацювання запобіжників.
    """
    detector.reset()
    logger = TrajectoryLogger()
    started = time.monotonic()

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    step_count, stop_reason = 0, None

    # stream(updates) віддає по одному оновленню на кожен відпрацьований вузол —
    # це і є готова траєкторія, логувати всередині вузлів не треба.
    stream = app.stream(
        {
            "messages": messages,
            "step_count": 0,
            "max_steps": max_steps,
            "deadline": started + timeout_s,
            "stop_reason": None,
        },
        # Технічний ліміт LangGraph: свій, вищий за max_steps, щоб спрацював
        # саме наш запобіжник, а не внутрішній GraphRecursionError.
        {"recursion_limit": max_steps * 2 + 10},
        stream_mode="updates",
    )
    for update in stream:
        for node, patch in update.items():
            step_count = patch.get("step_count", step_count)
            stop_reason = patch.get("stop_reason") or stop_reason
            new_messages = patch.get("messages", [])
            output, names = _describe(new_messages)
            logger.log_step(step_count, node, str(messages[-1].content), output, names)
            messages += new_messages

    # ── Structured output (Завдання 2) ───────────────────────────────────────
    text = str(messages[-1].content)
    tool_calls = [c["name"] for m in messages for c in (getattr(m, "tool_calls", None) or [])]
    observations = [str(m.content) for m in messages if m.__class__.__name__ == "ToolMessage"]
    final = build_final_answer(text, sorted(set(tool_calls)), observations, stop_reason)
    logger.log_step(step_count, "finalize", text, str(final.model_dump()))
    elapsed_ms = int((time.monotonic() - started) * 1000)

    if trajectory_path:
        logger.save(trajectory_path, query=query, stop_reason=stop_reason)

    return {
        "query": query,
        "answer": final.answer,
        "confidence": final.confidence,
        "sources": final.sources,
        "steps": step_count,
        "tool_calls": tool_calls,
        "stop_reason": stop_reason,
        "elapsed_ms": elapsed_ms,
        "trajectory": logger.as_dict(query, stop_reason),
    }


if __name__ == "__main__":
    import sys

    user_query = " ".join(sys.argv[1:]) or "Яка погода у Львові на 3 дні?"
    outcome = run_agent(user_query, trajectory_path="trajectory.json")
    print("\n=== Відповідь ===")
    print(outcome["answer"])
    print(f"\nКроків: {outcome['steps']} | Інструменти: {outcome['tool_calls']}")
    print(f"Впевненість: {outcome['confidence']} | Джерела: {outcome['sources']}")
    print(f"Причина зупинки: {outcome['stop_reason']} | Час: {outcome['elapsed_ms']} мс")
