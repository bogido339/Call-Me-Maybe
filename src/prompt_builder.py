from typing import Any


from typing import Any

def build_function_name_prompt(
    user_question: str,
    functions: list[dict[str, Any]],
) -> str:
    """Build a prompt that asks the model to pick a function name.

    Lists all available functions with their parameter types and instructs
    the model to respond with only the function name.
    """
    functions_list = ""

    functions_list = "\n".join(
        f'[{fn["name"]}: {fn["description"]}]'
        for fn in functions
    )

    return f"""You are a strict function-calling assistant

Available functions:
{functions_list}

User question:
{user_question}

function name:
"""


def build_function_args_prompt(
    user_question: str,
    selected_function: dict[str, Any],
) -> str:
    """Build a prompt that asks the model to extract argument values.

    Includes the function name, expected parameter types as hints, and
    the user request. The output is expected to continue as a JSON object.
    """
    arg_hints = ", ".join(
        f'"{param_name}": <{param_info["type"]}>'
        for param_name, param_info
        in selected_function["parameters"].items()
    )

    return (
        f"You are a professional Function Argument Extraction Engine.\n\n"
        f"Function:\n{selected_function['name']}\n\n"
        f"Expected Parameters:\n{{{arg_hints}}}\n\n"
        f"User Request:\n{user_question}\n\n"
        f"Parameters:\n"
    )
