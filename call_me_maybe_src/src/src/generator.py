from typing import Any


def generate_function_name(
    model: Any,
    prompt: str,
    functions: list[dict[str, Any]],
) -> str:
    """Generate a valid function name using constrained token masking.

    At each decoding step, only token IDs that belong to known function
    names are kept. Generation stops as soon as the output exactly matches
    one of the allowed function names.
    """
    input_ids: list[int] = model.encode(prompt).tolist()[0]
    generated_text = ""

    allowed_functions: list[str] = [
        str(fn["name"]) for fn in functions if "name" in fn
    ]

    allowed_tokens: set[int] = {
        token_id
        for fn in allowed_functions
        for token_id in model.encode(fn).tolist()[0]
    }

    for _ in range(100):
        logits: list[float] = model.get_logits_from_input_ids(input_ids)

        masked_logits: list[float] = [-float("inf")] * len(logits)

        for token_id in allowed_tokens:
            masked_logits[token_id] = logits[token_id]

        next_token: int = masked_logits.index(max(masked_logits))
        input_ids.append(next_token)

        token_text: str = model.decode(next_token)
        generated_text += token_text

        if generated_text in allowed_functions:
            break

    return generated_text.strip()


def generate_function_args(
    model: Any,
    prompt: str,
    types: str,
) -> str:
    """Generate the value for a single function argument.

    For numeric and boolean types, token masking restricts generation to
    digit characters only. For string types, generation is unconstrained.
    Stops at any EOS token (comma, brace, newline, or model-specific tokens).
    """
    input_ids: list[int] = model.encode(prompt).tolist()[0]
    generated_text = ""

    eos_tokens = {
        ",", "\n", "}", "</s>", "<|end|>", "<|im_end|>", "<|endoftext|>"
    }

    allowed_characters = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '-', '.', '{', '}', ",", "\n",
    ]

    allowed_tokens: set[int] = {
        token_id
        for char in allowed_characters
        for token_id in model.encode(char).tolist()[0]
    }

    for _ in range(100):
        logits: list[float] = model.get_logits_from_input_ids(input_ids)

        if types in ("number", "integer", "boolean"):
            masked_logits: list[float] = [-float("inf")] * len(logits)

            for token_id in allowed_tokens:
                masked_logits[token_id] = logits[token_id]

            next_token: int = masked_logits.index(max(masked_logits))
        else:
            next_token = logits.index(max(logits))

        input_ids.append(next_token)
        token_text: str = model.decode(next_token)

        if any(eos in token_text for eos in eos_tokens):
            for eos in eos_tokens:
                if eos in token_text:
                    token_text = token_text.split(eos)[0]
                    break
            generated_text += token_text
            break

        generated_text += token_text

    return generated_text.strip()
