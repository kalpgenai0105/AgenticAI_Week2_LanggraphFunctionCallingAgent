"""

----------------------------------------------------------
Kalpesh's LangGraph Function Calling Agent Demo
Week 2 Homework: LangGraph Function Calling Agent
----------------------------------------------------------

LangGraph ReAct agent with Calculator and Weather tools, powered by a free Groq model.

Written against:
    langgraph>=0.3          langchain-groq>=0.3
    langchain-core>=0.3     python-dotenv>=1.0
    requests>=2.31

Install:
    pip install "langgraph>=0.3" "langchain-groq>=0.3" "langchain-core>=0.3" \
                python-dotenv requests
"""

import ast
import operator
import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load GROQ_API_KEY and OPENWEATHER_API_KEY from a .env file if present.
# Keys already set in the environment take precedence (dotenv does not overwrite).
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is not set.\n"
        "  → Get a free key at https://console.groq.com\n"
        "  → Add it to a .env file: GROQ_API_KEY=gsk_..."
    )
if not OPENWEATHER_API_KEY:
    raise EnvironmentError(
        "OPENWEATHER_API_KEY is not set.\n"
        "  → Get a free key at https://openweathermap.org/api\n"
        "  → Add it to a .env file: OPENWEATHER_API_KEY=..."
    )

# llama3-70b-8192: stable free-tier model with solid function-calling support.
# Other options: "llama-3.1-70b-versatile", "llama3-8b-8192" (faster, lighter)
MODEL_NAME = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# Tool 1 — Safe calculator
# ---------------------------------------------------------------------------

# Whitelist of AST node types we permit during expression evaluation.
# This prevents arbitrary code execution that a raw eval() would allow.
_SAFE_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
}

_OP_MAP = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.UAdd:     operator.pos,
    ast.USub:     operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if type(node) not in _SAFE_NODES:
        raise ValueError(f"Disallowed operation: {type(node).__name__}")

    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        return float(node.value)

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        fn = _OP_MAP.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return fn(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        fn = _OP_MAP.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return fn(operand)

    raise ValueError(f"Unexpected node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    Safely evaluate an arithmetic expression and return the numeric result.

    Supports: +, -, *, / (true division), // (floor division), % (modulo),
    ** (exponentiation), unary +/-, and nested parentheses.
    Does NOT support variables, function calls, imports, or string operations.

    Args:
        expression: A valid arithmetic expression string, e.g. "(3 + 5) * 2 ** 3".

    Returns:
        The computed result as a string, or an error message string if the
        expression is invalid, disallowed, or causes a division by zero.
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
        # Show an integer when there is no fractional part
        return str(int(result)) if result == int(result) else str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, TypeError, SyntaxError) as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Tool 2 — OpenWeather current conditions
# ---------------------------------------------------------------------------

_OW_URL = "https://api.openweathermap.org/data/2.5/weather"


@tool
def get_weather(city: str) -> str:
    """
    Fetch current weather conditions for a city via the OpenWeather API.

    Returns temperature (°C and °F), feels-like temperature, humidity,
    sky/weather description, wind speed, and visibility.

    Args:
        city: City name, optionally with ISO country code, e.g. "Paris" or "London,GB".
              Adding the country code helps resolve ambiguous city names.

    Returns:
        A formatted multi-line weather summary string.
        On failure, returns a descriptive error string (unknown city, bad API key,
        network error, or unexpected HTTP status) instead of raising an exception,
        so the agent can relay the problem back to the user.
    """
    try:
        resp = requests.get(
            _OW_URL,
            params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"},
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        return "Error: Could not reach the OpenWeather API. Check your internet connection."
    except requests.exceptions.Timeout:
        return "Error: The OpenWeather API request timed out."

    if resp.status_code == 401:
        return "Error: Invalid or missing OpenWeather API key."
    if resp.status_code == 404:
        return (
            f"Error: City '{city}' not found. "
            "Try including a country code, e.g. 'Springfield,US'."
        )
    if resp.status_code != 200:
        return f"Error: OpenWeather returned HTTP {resp.status_code}: {resp.text}"

    d = resp.json()
    temp_c      = d["main"]["temp"]
    temp_f      = temp_c * 9 / 5 + 32
    feels_c     = d["main"]["feels_like"]
    humidity    = d["main"]["humidity"]
    description = d["weather"][0]["description"].capitalize()
    wind_kmh    = d["wind"]["speed"] * 3.6
    vis_m       = d.get("visibility")
    vis_str     = f"{vis_m / 1000:.1f} km" if vis_m is not None else "N/A"
    name        = d.get("name", city)
    country     = d.get("sys", {}).get("country", "")

    location = f"{name}, {country}" if country else name
    return (
        f"Weather in {location}:\n"
        f"  Conditions : {description}\n"
        f"  Temperature: {temp_c:.1f}°C ({temp_f:.1f}°F)\n"
        f"  Feels like : {feels_c:.1f}°C\n"
        f"  Humidity   : {humidity}%\n"
        f"  Wind speed : {wind_kmh:.1f} km/h\n"
        f"  Visibility : {vis_str}"
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_tools = [calculator, get_weather]

# temperature=0 for deterministic, reproducible tool-call decisions
_llm = ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY, temperature=0)

# create_react_agent wires up: LLM node (with tools bound) → ToolNode → loop until
# the model produces a final answer with no pending tool calls.
agent = create_react_agent(model=_llm, tools=_tools)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_query(query: str) -> None:
    """
    Send one query to the agent, streaming state updates so every tool call
    and result is printed as it happens before the final answer appears.
    """
    print(f"\n{'=' * 60}")
    print(f"User: {query}")
    print("=" * 60)

    # stream_mode="updates" yields {node_name: state_delta} after each graph node.
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="updates",
    ):
        for _node, output in step.items():
            for msg in output.get("messages", []):
                msg_type = getattr(msg, "type", type(msg).__name__)

                if msg_type == "ai":
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"\n[Tool call]   {tc['name']}({tc['args']})")
                    elif msg.content:
                        print(f"\nAssistant: {msg.content}")

                elif msg_type == "tool":
                    print(f"[Tool result] {msg.name} → {msg.content}")


# ---------------------------------------------------------------------------
# Demo queries + interactive loop
# ---------------------------------------------------------------------------

_DEMO_QUERIES = [
    # Calculator only
    "What is (144 / 12) + 7 * 3?",
    # Weather only
    "What's the current weather in Tokyo?",
    # Both tools in sequence: get the temperature, then do arithmetic on it
    "What's the temperature in Paris right now, and what is that Celsius value "
    "converted to Fahrenheit using the formula (C * 9/5) + 32?",
]


if __name__ == "__main__":
    print("LangGraph ReAct Agent — Groq / Calculator / Weather")
    print(f"Model : {MODEL_NAME}")

    print("\n--- Demo queries ---")
    for q in _DEMO_QUERIES:
        run_query(q)

    print(f"\n{'=' * 60}")
    print("Interactive mode  (type 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if user_input:
            run_query(user_input)
