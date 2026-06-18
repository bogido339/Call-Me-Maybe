from llm_sdk.llm_sdk import Small_LLM_Model
import json


def load_functions():
    with open("data/input/functions_definition.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompts():
    with open("data/input/function_calling_tests.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_function_name_prompt(user_question, functions):
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


def build_function_args_prompt(user_question, selected_function):
    return f"""
You are a strict argument extraction assistant.

Function:
{selected_function["name"]}

Parameters:
{json.dumps(selected_function["parameters"], indent=2)}

User question:
{user_question}

Respond ONLY with a JSON object containing the arguments.
"""

def generate_text(model, prompt, max_tokens=100):
    input_ids = model.encode(prompt).tolist()[0]

    generated_text = ""

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)

        next_token = logits.index(max(logits))

        input_ids.append(next_token)

        token_text = model.decode(next_token)

        if token_text in ["", "\n", "</s>", "<|end|>", "<|im_end|>", "<|endoftext|>"]:
            break

        generated_text += token_text

    return generated_text.strip()


def select_function(model, user_question, functions):
    prompt = build_function_name_prompt(user_question, functions)

    function_name = generate_text(model, prompt)

    return function_name

def extract_arguments(model, user_question, function_name, functions):
    selected_function = None

    for fn in functions:
        if fn["name"] == function_name:
            selected_function = fn
            break

    if selected_function is None:
        return {}

    prompt = build_function_args_prompt(user_question, selected_function)

    response = generate_text(model, prompt)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}

def build_json_result(function_name, function_arguments):
    return {
        "function_name": function_name,
        "arguments": function_arguments
    }


def process_question(model, user_question, functions):
    function_name = select_function(model, user_question,functions)

    function_arguments = extract_arguments(model, user_question, function_name, functions)

    result = build_json_result(function_name, function_arguments)

    return result

def main():
    model = Small_LLM_Model()

    functions = load_functions()

    prompts = load_prompts()

    for item in prompts:
        user_question = item["prompt"]

        result = process_question(model, user_question, functions)

        print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
