from llm_sdk.llm_sdk import Small_LLM_Model
import json


def build_prompt(user_question: str, functions: list) -> str:
    functions_list = ""
    for fn in functions:
        args = ", ".join(
            f"{param_name}: {param_info['type']}"
            for param_name, param_info in fn["parameters"].items()
        )
        functions_list += f"- {fn['name']}({args}) → {fn['description']}\n"

    return f"""You are a function-calling assistant.
Available functions:
{functions_list}
User question: {user_question}

Respond with the correct function name and arguments as JSON."""


def main():
    Amadeus = Small_LLM_Model()

    with open("data/input/functions_definition.json", "r", encoding="utf-8") as f:
        functions = json.load(f)

    with open("data/input/function_calling_tests.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    for user_question in prompts:
        prompt = build_prompt(user_question.get("prompt"), functions)
        input_ids = Amadeus.encode(prompt).tolist()[0]

        generated = []
        for _ in range(100):
            logits = Amadeus.get_logits_from_input_ids(input_ids)
            next_token = logits.index(max(logits))
            input_ids.append(next_token)
            generated.append(next_token)
            token_text = Amadeus.decode(next_token)
            if token_text in ["</s>", "<|end|>", "<|im_end|>", "<|endoftext|>"]:
                break

        full_output = Amadeus.decode(generated)
        print(f"Q: {user_question.get('prompt')}")
        print(f"A: {full_output}")
        print()


if __name__ == "__main__":
    main()