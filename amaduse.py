from llm_sdk.llm_sdk import Small_LLM_Model


def main():
    model = Small_LLM_Model()

    prompt = "What is the color of the sky?"

    input_ids = model.encode(prompt).tolist()[0]

    allowed_tokens = []

    choices = ["Blue", "Light blue", "Gray", "Orange", "Black"]

    for choice in choices:
        token_ids = model.encode(choice).tolist()[0]
        allowed_tokens.append(token_ids[0])

    result = ""

    for _ in range(100):
        logits = model.get_logits_from_input_ids(input_ids)

        masked_logits = [-float("inf")] * len(logits)

        for token_id in allowed_tokens:
            masked_logits[token_id] = logits[token_id]

        next_token = masked_logits.index(max(masked_logits))

        token_text = model.decode([next_token])
        if token_text in ["", "\n"]:
            break

        result += token_text
        input_ids.append(next_token)

    print(result)


if __name__ == "__main__":
    main()