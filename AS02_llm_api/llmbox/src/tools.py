'''
    LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
    Copyright (C) 2026  Sara Kingsley

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

"""
tools.py

Example tool implementations for mode=tool_calling. When a model's chat
template doesn't have native structured tool-calling support and instead
uses a convention like Phi-4-mini-instruct's <|tool|>...<|/tool|> system
prompt + "functools[...]" response marker (see
GenerationManager.build_functools_system_prompt / parse_tool_calls /
run_tool_turn in generation.py), the model's requested tool name is looked
up here and actually executed locally.

Swap these out for whatever you actually want the model to be able to
call. Each tool needs:
  1. An entry under `tool_calling.tools` in your config (name, description,
     and a JSON-schema `parameters` block) -- that's what gets shown to the
     model.
  2. A matching Python callable registered here under the same name --
     that's what actually runs when the model asks for it.
"""

TOOL_REGISTRY = {}

def register_tool(name):
    """Decorator: register a callable under `name` in TOOL_REGISTRY, the
    lookup table GenerationManager.run_tool_turn / run_generic_tool_turn use
    to actually execute a tool call the model requests."""
    def decorator(func):
        TOOL_REGISTRY[name] = func
        return func
    return decorator

'''
@staticmethod
def register_tool(name):
    def decorator(func):
        TOOL_REGISTRY[name] = func
        return func
    return decorator
'''

@staticmethod
@register_tool("get_current_weather")
def get_current_weather(location, unit="celsius"):
    # Stub data -- replace this body with a real weather API call.
    fake_data = {"Paris, France": 18, "New York, USA": 22, "Tokyo, Japan": 27}
    temp_c = fake_data.get(location, 20)
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32, 1)
    return {"location": location, "temperature": temp, "unit": unit}

@register_tool("get_current_weather")
def get_current_weather(location, unit="celsius"):
    # Stub data -- replace this body with a real weather API call.
    fake_data = {"Paris, France": 18, "New York, USA": 22, "Tokyo, Japan": 27}
    temp_c = fake_data.get(location, 20)
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32, 1)
    return {"location": location, "temperature": temp, "unit": unit}


@register_tool("calculate")
def calculate(expression):
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return {"error": "Expression contains disallowed characters."}
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as exc:  # noqa: BLE001 - want to surface any eval error to the model
        return {"error": str(exc)}
