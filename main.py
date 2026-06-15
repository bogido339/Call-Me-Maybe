from llm_sdk.llm_sdk import Small_LLM_Model
import json


def score_candidate(model, question, function_name):
    prompt = f"""
Question: {question}

Selected function: {function_name}
"""

    input_ids = model.encode(prompt).tolist()[0]

    score = 0.0

    for _ in range(20):
        logits = model.get_logits_from_input_ids(input_ids)

        # Use the highest logit as a simple score
        score += max(logits)

        next_token = logits.index(max(logits))
        input_ids.append(next_token)

    return score


def main():
    model = Small_LLM_Model()

    with open(
        "data/input/functions_definition.json",
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    function_names = [
        item["name"]
        for item in data
        if "name" in item
    ]

    with open("data/input/function_calling_tests.json", "r", encoding="utf-8") as f:
      promts = json.load(f)

    for p in promts:

      question = f"Question: {p.get("prompt")}"

      # question = "What is the sum of 2 and 3?"

      best_function = None
      best_score = float("-inf")

      for fn in function_names:
          score = score_candidate(model, question, fn)

          if score > best_score:
              best_score = score
              best_function = fn

      print("Selected function:", best_function)


if __name__ == "__main__":
    main()