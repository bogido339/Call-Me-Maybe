import json
import argparse
import os
import sys
from argparse import Namespace

from llm_sdk.llm_sdk import Small_LLM_Model

from src.data_loader import load_functions, load_prompts
from src.processor import process_question


def parse_args() -> Namespace:
    """Parse command-line arguments for input/output file paths."""
    parser = argparse.ArgumentParser(description="Function Calling Pipeline")
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to the output JSON file",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load inputs, run the pipeline, write results to output."""
    args = parse_args()

    model = Small_LLM_Model()
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    results = []

    for item in prompts:
        user_question: str = item["prompt"]
        output = process_question(model, user_question, functions)
      
        results.append(output)
        print(json.dumps(output, indent=4, ensure_ascii=False))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
