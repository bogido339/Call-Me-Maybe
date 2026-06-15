from llm_sdk.llm_sdk import Small_LLM_Model
import json


def main():
    model = Small_LLM_Model()

    with open("data/input/functions_definition.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    function_names = [item["name"] for item in data if "name" in item]

    with open("data/input/function_calling_tests.json", "r", encoding="utf-8") as f:
        promts = json.load(f)

    for p in promts:
        prompt = f"""
    Question: {p.get("prompt")}

    Available functions:
    {", ".join(function_names)}

    Choose the best function name from the list above.
    Return only the function name.
    """

        input_ids = model.encode(prompt).tolist()[0]

        output = []

        max_tokens = 20

        for _ in range(max_tokens):
            logits = model.get_logits_from_input_ids(input_ids)

            next_token = logits.index(max(logits))
            token_text = model.decode(next_token)

            output.append(token_text)
            input_ids.append(next_token)

            if token_text in ["\n", ".", "</s>"]:
                break

        result = "".join(output).strip()
        print("Selected function:", result)


if __name__ == "__main__":
    main()