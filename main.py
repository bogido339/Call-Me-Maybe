from llm_sdk.llm_sdk import Small_LLM_Model
import json
from typing import List


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
    arg_hints = ", ".join(
        f'"{param_name}": <{param_info["type"]}>'
        for param_name, param_info in selected_function["parameters"].items()
    )

    return f"""You are a strict argument extraction assistant.

Function: {selected_function["name"]}

Extract the arguments from the user question and return ONLY a JSON object.
Do NOT include explanations, markdown, or extra text.

Expected format:
{{{arg_hints}}}

User question: {user_question}

JSON:"""


def generate_function_name(model, prompt: str, functions: List):
    input_ids = model.encode(prompt).tolist()[0]

    generated_text = ""

    allowed_functions = [fn.get("name") for fn in functions]

    allowed_tokens = {
        token_id
        for fn in allowed_functions
        for token_id in model.encode(fn).tolist()[0]
    }


    for _ in range(100):
        logits = model.get_logits_from_input_ids(input_ids)

        masked_logits = [-float("inf")] * len(logits)

        for token_id in allowed_tokens:
            masked_logits[token_id] = logits[token_id]

        next_token = masked_logits.index(max(masked_logits))

        input_ids.append(next_token)

        token_text = model.decode(next_token)

        generated_text += token_text

        if generated_text in allowed_functions:
            break

    return generated_text.strip()

def generate_function_args(model, prompt, function):
    input_ids = model.encode(prompt).tolist()[0]

    generated_text = ""

    EOS_TOKENS = {"", "", "\n", "</s>", "<|end|>", "<|im_end|>", "<|endoftext|>"}

    for _ in range(100):
        logits = model.get_logits_from_input_ids(input_ids)

        next_token = logits.index(max(logits))

        input_ids.append(next_token)

        token_text = model.decode(next_token)

        if token_text in EOS_TOKENS:
            break

        generated_text += token_text

        if generated_text.strip().startswith("{") and generated_text.strip().endswith("}"):
            try:
                json.loads(generated_text.strip())
                break
            except json.JSONDecodeError:
                pass

    return generated_text.strip()


def select_function(model, user_question, functions):
    prompt = build_function_name_prompt(user_question, functions)

    function_name = generate_function_name(model, prompt, functions)

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

    response = generate_function_args(model, prompt, selected_function)

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

    with open("output.json", "w", encoding="utf-8") as f:
        for item in prompts:
            user_question = item["prompt"]

            result = process_question(model, user_question, functions)

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
