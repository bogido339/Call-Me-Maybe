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

    return f"""You are a professional Function Argument Extraction Engine.

Function:
{selected_function["name"]}

Expected Parameters:
{{{arg_hints}}}

User Request:
{user_question}

Parameters:\n"""

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

def generate_function_args(model, prompt, types):
    input_ids = model.encode(prompt).tolist()[0]
    generated_text = ""
    
    EOS_TOKENS = {",", "\n", "}", "</s>", "<|end|>", "<|im_end|>", "<|endoftext|>"}

    allowed_characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', '{', '}', ",", "\n"]

    allowed_tokens = {
        token_id
        for char in allowed_characters
        for token_id in model.encode(char).tolist()[0]
    }

    for _ in range(100):
        logits = model.get_logits_from_input_ids(input_ids)

        if types == "number" or types == "integer" or types == "boolean":
            masked_logits = [-float("inf")] * len(logits)

            for token_id in allowed_tokens:
                masked_logits[token_id] = logits[token_id]

            next_token = masked_logits.index(max(masked_logits))
        else:
            next_token = logits.index(max(logits))
            
        input_ids.append(next_token)
        
        token_text = model.decode(next_token)

        if any(eos in token_text for eos in EOS_TOKENS):
            for eos in EOS_TOKENS:
                if eos in token_text:
                    token_text = token_text.split(eos)[0]
                    break
            
            generated_text += token_text
            break

        generated_text += token_text

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
    
    prompt += " {" 
    res = {}
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
        
        if i < len(element) - 1:
            prompt += ", "

    return res

def build_json_result(prompt, function_name, function_arguments):
    return {
        "prompt": prompt,
        "name": function_name,
        "parameters": function_arguments
    }


def process_question(model, user_question, functions):
    function_name = select_function(model, user_question,functions)

    function_arguments = extract_arguments(model, user_question, function_name, functions)

    result = build_json_result(user_question, function_name, function_arguments)

    return result

def main():
    model = Small_LLM_Model()

    functions = load_functions()
    prompts = load_prompts()

    results = []

    for item in prompts:
        user_question = item["prompt"]

        output = process_question(model, user_question, functions)
        results.append(output)

        print(json.dumps(output, indent=4, ensure_ascii=False))

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
