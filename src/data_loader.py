import json
import os
from typing import Any


class ParserError(Exception):
    pass


VALID_TYPES = {"string", "number", "integer", "boolean"}


def load_functions(
    path: str = "data/input/functions_definition.json",
) -> list[dict[str, Any]]:
    """Load and validate the functions definition JSON file."""
    if not path or not isinstance(path, str):
        raise ParserError(f"Invalid path provided: {repr(path)}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Functions definition file not found: '{path}'"
        )

    if not os.path.isfile(path):
        raise IsADirectoryError(
            f"Expected a file but got a directory: '{path}'"
        )

    if not os.access(path, os.R_OK):
        raise PermissionError(
            f"No read permission for file: '{path}'"
        )

    if os.path.getsize(path) == 0:
        raise ParserError(
            f"Functions definition file is empty: '{path}'"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ParserError(
            f"Invalid JSON in functions definition file '{path}': {e}"
        )
    except UnicodeDecodeError as e:
        raise ParserError(
            f"Encoding error in file '{path}' (expected UTF-8): {e}"
        )
    except OSError as e:
        raise OSError(f"Failed to read file '{path}': {e}")

    if not isinstance(data, list):
        raise TypeError(
            f"Functions definition must be a JSON array, "
            f"got: {type(data).__name__}"
        )

    if len(data) == 0:
        raise ParserError(
            "Functions definition list is empty — no functions to call."
        )

    for i, fn in enumerate(data):
        if not isinstance(fn, dict):
            raise TypeError(
                f"Function at index {i} must be a JSON object, "
                f"got: {type(fn).__name__}"
            )

        for key in ("name", "description", "parameters", "returns"):
            if key not in fn:
                raise KeyError(
                    f"Function at index {i} is missing required key: '{key}'"
                )

        if not isinstance(fn["name"], str) or not fn["name"].strip():
            raise ParserError(
                f"Function at index {i} has an invalid "
                f"'name': {repr(fn['name'])}"
            )

        if (
            not isinstance(fn["description"], str)
            or not fn["description"].strip()
        ):
            raise ParserError(
                f"Function '{fn['name']}' has an invalid 'description'."
            )

        if not isinstance(fn["parameters"], dict):
            raise ParserError(
                f"Function '{fn['name']}': "
                f"'parameters' must be a JSON object."
            )

        for param_name, param_info in fn["parameters"].items():
            if not isinstance(param_info, dict):
                raise ParserError(
                    f"Function '{fn['name']}': parameter '{param_name}' "
                    f"must be a JSON object."
                )
            if "type" not in param_info:
                raise KeyError(
                    f"Function '{fn['name']}': parameter '{param_name}' "
                    f"is missing 'type'."
                )
            if param_info["type"] not in VALID_TYPES:
                raise ParserError(
                    f"Function '{fn['name']}': parameter '{param_name}' "
                    f"has unsupported type '{param_info['type']}'. "
                    f"Must be one of: {VALID_TYPES}"
                )

        if (
            not isinstance(fn["returns"], dict)
            or "type" not in fn["returns"]
        ):
            raise ParserError(
                f"Function '{fn['name']}': 'returns' must have a 'type' field."
            )

    return data


def load_prompts(
    path: str = "data/input/function_calling_tests.json",
) -> list[dict[str, Any]]:
    """Load and validate the prompts JSON file."""
    if not path or not isinstance(path, str):
        raise ParserError(f"Invalid path provided: {repr(path)}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompts file not found: '{path}'")

    if not os.path.isfile(path):
        raise IsADirectoryError(
            f"Expected a file but got a directory: '{path}'"
        )

    if not os.access(path, os.R_OK):
        raise PermissionError(f"No read permission for file: '{path}'")

    if os.path.getsize(path) == 0:
        raise ParserError(f"Prompts file is empty: '{path}'")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ParserError(
            f"Invalid JSON in prompts file '{path}': {e}"
        )
    except UnicodeDecodeError as e:
        raise ParserError(
            f"Encoding error in file '{path}' (expected UTF-8): {e}"
        )
    except OSError as e:
        raise OSError(f"Failed to read file '{path}': {e}")

    if not isinstance(data, list):
        raise TypeError(
            f"Prompts file must be a JSON array, got: {type(data).__name__}"
        )

    if len(data) == 0:
        raise ParserError("Prompts list is empty — nothing to process.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(
                f"Prompt at index {i} must be a JSON object, "
                f"got: {type(item).__name__}"
            )
        if "prompt" not in item:
            raise KeyError(
                f"Prompt at index {i} is missing required key: 'prompt'"
            )

        if (
            not isinstance(item["prompt"], str)
            or not item["prompt"].strip()
        ):
            raise ParserError(
                f"Prompt at index {i} has an empty or invalid 'prompt' value."
            )

    return data
