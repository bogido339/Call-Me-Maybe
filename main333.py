from llm_sdk.llm_sdk import Small_LLM_Model
import json
from typing import List, Dict, Any


def load_functions():
    with open("data/input/functions_definition.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts():
    with open("data/input/function_calling_tests.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_function_name_prompt(user_question: str, functions: List[dict]) -> str:
    functions_list = ""

    for fn in functions:
        args = ", ".join(
            f'"{param_name}": {param_info["type"]}'
            for param_name, param_info in fn["parameters"].items()
        )

        functions_list += f'- {fn["name"]}: {{{args}}}\n'

    return f"""
You are a strict function-calling assistant.

Available functions:
{functions_list}

User question:
{user_question}

Respond ONLY with the function name.
"""


def build_function_args_prompt(
    user_question: str,
    function_name: str,
    arg_name: str
) -> str:
    return f"""
You are a strict argument extraction assistant.

Function:
{function_name}

Parameter:
{arg_name}

User question:
{user_question}

Respond ONLY with the value of the parameter.
"""


def generate_function_name(
    model,
    prompt: str,
    functions: List[dict]
) -> str:

    input_ids = model.encode(prompt).tolist()[0]

    generated_text = ""

    allowed_functions = [fn["name"] for fn in functions]

    allowed_tokens = set()

    for fn_name in allowed_functions:
        token_ids = model.encode(fn_name).tolist()[0]
        allowed_tokens.update(token_ids)

    for _ in range(50):
        logits = model.get_logits_from_input_ids(input_ids)

        masked_logits = [-float("inf")] * len(logits)

        for token_id in allowed_tokens:
            if token_id < len(masked_logits):
                masked_logits[token_id] = logits[token_id]

        next_token = masked_logits.index(max(masked_logits))

        input_ids.append(next_token)

        token_text = model.decode(next_token)
        generated_text += token_text

        cleaned = generated_text.strip()

        if cleaned in allowed_functions:
            return cleaned

    return generated_text.strip()


def generate_function_args(
    model,
    prompt: str
) -> str:

    input_ids = model.encode(prompt).tolist()[0]

    generated_text = ""

    eos_tokens = {
        "",
        "\n",
        "</s>",
        "<|end|>",
        "<|im_end|>",
        "<|endoftext|>",
    }

    for _ in range(50):
        logits = model.get_logits_from_input_ids(input_ids)

        next_token = logits.index(max(logits))

        input_ids.append(next_token)

        token_text = model.decode(next_token)

        if token_text in eos_tokens:
            break

        generated_text += token_text

    return generated_text.strip()


def select_function(
    model,
    user_question: str,
    functions: List[dict]
) -> str:

    prompt = build_function_name_prompt(user_question, functions)

    return generate_function_name(model, prompt, functions)


def extract_arguments(
    model,
    user_question: str,
    function_name: str,
    functions: List[dict]
) -> Dict[str, Any]:

    selected_function = None

    for fn in functions:
        if fn["name"] == function_name:
            selected_function = fn
            break

    if selected_function is None:
        return {}

    arguments = {}

    for arg_name in selected_function["parameters"]:
        prompt = build_function_args_prompt(
            user_question,
            function_name,
            arg_name,
        )

        value = generate_function_args(model, prompt)

        arguments[arg_name] = value

    return arguments


def build_json_result(
    function_name: str,
    function_arguments: Dict[str, Any]
) -> dict:

    return {
        "function_name": function_name,
        "arguments": function_arguments,
    }


def process_question(
    model,
    user_question: str,
    functions: List[dict]
) -> dict:

    function_name = select_function(
        model,
        user_question,
        functions,
    )

    function_arguments = extract_arguments(
        model,
        user_question,
        function_name,
        functions,
    )

    return build_json_result(
        function_name,
        function_arguments,
    )


def main():
    model = Small_LLM_Model()

    functions = load_functions()

    prompts = load_prompts()

    for item in prompts:
        user_question = item["prompt"]

        result = process_question(
            model,
            user_question,
            functions,
        )

        print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()