from typing import Any

from src.prompt_builder import (
    build_function_name_prompt,
    build_function_args_prompt,
)
from src.generator import generate_function_name, generate_function_args


def select_function(
    model: Any,
    user_question: str,
    functions: list[dict[str, Any]],
) -> str:
    """Select the most appropriate function name for a given user question.

    Builds a name-selection prompt and runs constrained decoding to pick
    exactly one valid function name from the available list.
    """
    prompt = build_function_name_prompt(user_question, functions)
    function_name: str = generate_function_name(model, prompt, functions)
    return str(function_name)


def extract_arguments(
    model: Any,
    user_question: str,
    function_name: str,
    functions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract and type-cast all arguments for the selected function.

    Iterates over each expected parameter, runs constrained decoding for
    its value, strips whitespace/quotes, and casts to the correct Python type.
    Returns an empty dict if the function name is not found.
    """
    selected_function: dict[str, Any] | None = None

    for fn in functions:
        if fn["name"] == function_name:
            selected_function = fn
            break

    if selected_function is None:
        return {}

    prompt = build_function_args_prompt(user_question, selected_function)
    prompt += " {"
    res: dict[str, Any] = {}
    items = list(selected_function["parameters"].items())


    for i, element in enumerate(items):
        name, types = element
        prompt += f'"{name}":'

        response = generate_function_args(model, prompt, types["type"])
        response = response.strip()
        response = response.rstrip(",} \n")
        response = response.strip("\"'")

        if types["type"] == "number":
            res[name] = float(response)
        elif types["type"] == "integer":
            res[name] = int(response)
        elif types["type"] == "boolean":
            res[name] = bool(response)
        else:
            res[name] = response

        prompt += f' "{response}"'

        if i < len(items) - 1:
            prompt += ", "

    return res


def build_json_result(
    prompt: str,
    function_name: str,
    function_arguments: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the final output dict from prompt, name, and arguments."""
    return {
        "prompt": prompt,
        "name": function_name,
        "parameters": function_arguments,
    }


def process_question(
    model: Any,
    user_question: str,
    functions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run the full pipeline: select function, extract args, build result."""
    function_name = select_function(model, user_question, functions)
    function_arguments = extract_arguments(
        model, user_question, function_name, functions
    )
    result = build_json_result(
        user_question, function_name, function_arguments
    )
    return result
