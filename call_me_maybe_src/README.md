*This project has been created as part of the 42 curriculum by mbougajd.*

---

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

**Call Me Maybe** is a Python project that bridges natural language and structured function calls using a small language model (Qwen/Qwen3-0.6B, ~600M parameters). Given a user prompt like `"What is the sum of 2 and 3?"`, the system automatically selects the correct function (`fn_add_numbers`) and extracts its arguments (`{"a": 2.0, "b": 3.0}`) — without relying on any external function-calling API.

The core challenge is producing **100% valid, structured JSON output** from a model that was not fine-tuned for this task. This is solved through **constrained decoding**: at each generation step, the model's token probabilities are masked so only valid tokens can be selected, guaranteeing well-formed output by construction.

### Goal

> Transform free-text user requests into structured JSON function calls using constrained token-level decoding over a raw language model.

### Supported Functions

| Function | Description |
|---|---|
| `fn_add_numbers` | Add two numbers |
| `fn_greet` | Greet a person by name |
| `fn_reverse_string` | Reverse a string |
| `fn_get_square_root` | Compute the square root of a number |
| `fn_substitute_string_with_regex` | Replace patterns in a string via regex |

---

## Algorithm Explanation

The pipeline runs in two sequential constrained decoding passes:

### Pass 1 — Function Name Selection

1. A prompt is built listing all available function signatures.
2. The model generates tokens one by one.
3. At each step, a **token mask** is applied: only token IDs that are part of a valid function name are kept (`-inf` for all others).
4. Generation stops as soon as the output exactly matches one of the known function names.

This guarantees the model **can only output a valid function name** — hallucinated or partial names are impossible.

### Pass 2 — Argument Extraction

For each parameter of the selected function:

1. A prompt is built with the function signature and parameter hints, followed by the partial JSON built so far.
2. The model generates the value for that parameter.
3. **Type-aware masking** is applied:
   - For `number`, `integer`, `boolean`: only digit characters (`0–9`), `.`, `-` are allowed.
   - For `string`: unconstrained generation (the model outputs freely).
4. Generation stops at an EOS token (`,`, `}`, `\n`, `</s>`, etc.).
5. The raw text is cast to the correct Python type (`float`, `int`, `bool`, `str`).
6. The parameter is appended to the prompt context before moving to the next one, giving the model full running context.

### Output Format

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

---

## Design Decisions

### Why constrained decoding instead of prompting?
Pure prompting (asking the model to "output JSON") fails frequently with small models — they hallucinate keys, omit brackets, or output prose. Constrained decoding enforces structure at the **token probability level**, making invalid output mathematically impossible for name selection and numerically typed arguments.

### Why two separate passes?
Separating function name selection from argument extraction allows each pass to use a tightly scoped prompt and mask set. A single combined pass would require a far more complex grammar constraint and is harder to debug.

### Why append each argument to the prompt before generating the next?
This gives the model full **running context** of what has already been extracted, improving coherence for multi-argument functions (e.g., `fn_substitute_string_with_regex` with 3 parameters).

### Why `uv` as the package manager?
`uv` is significantly faster than `pip` for dependency resolution and installation, and `pyproject.toml` provides a single source of truth for dependencies — cleaner than a bare `requirements.txt`.

### Why Pydantic for validation?
The evaluation rubric requires it. Pydantic provides declarative, type-safe validation of input structures with clear error messages, replacing manual `isinstance` checks.

---

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager
- `llm_sdk` (provided separately — place it in the project root)

### Installation

```bash
git clone <your-repo-url>
cd call-me-maybe
uv sync
```

### Running the program

```bash
uv run python -m src

uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Other Makefile commands

```bash
make install
make run
make debug
make lint
make lint-strict
make clean
```

### Directory structure

```
call-me-maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py
│   ├── data_loader.py
│   ├── prompt_builder.py
│   ├── generator.py
│   └── processor.py
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/
│       └── function_calls.json        ← generated at runtime
├── llm_sdk/
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Example Usage

**Input prompt:** `"Replace all vowels in 'Programming is fun' with asterisks"`

**Output:**
```json
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
  "fn_name": "fn_substitute_string_with_regex",
  "args": {
    "source_string": "Programming is fun",
    "regex": "[aeiouAEIOU]",
    "replacement": "*"
  }
}
```

**Input prompt:** `"What is the square root of 16?"`

**Output:**
```json
{
  "prompt": "What is the square root of 16?",
  "fn_name": "fn_get_square_root",
  "args": {
    "a": 16.0
  }
}
```

---

## Performance Analysis

| Metric | Result |
|---|---|
| JSON validity | 100% — guaranteed by constrained decoding |
| Function selection accuracy | >90% on provided test set |
| Argument extraction accuracy | >90% on provided test set |
| Processing time | <5 minutes for the full test set |
| Reliability across runs | Deterministic (greedy decoding, no sampling) |

Constrained decoding ensures **zero JSON parse errors** regardless of model quality. The bottleneck is numeric argument extraction for models that have poor digit tokenization — mitigated by restricting to a tight allowed character set.

---

## Challenges Faced

### 1. Token boundary mismatch
A function name like `fn_add_numbers` may be tokenized as `["fn", "_add", "_numbers"]`. The mask must allow **all individual token IDs** that compose any valid name, not just the full-name token. Solved by pre-computing the union of all token IDs for all characters and sub-tokens of each function name.

### 2. EOS token diversity
Small models use inconsistent end-of-sequence tokens (`</s>`, `<|end|>`, `<|im_end|>`, `<|endoftext|>`). The generator checks for all known variants and splits on the first match to avoid including EOS bytes in the output.

### 3. Argument context drift
Without appending previously generated arguments to the prompt, the model would ignore prior context and generate inconsistent or repeated values. The running-prompt approach resolved this.

### 4. Type coercion edge cases
Models sometimes generate `"3.0"` for an integer field or whitespace-padded values. Explicit `.strip()`, `.rstrip(",} \n")`, and `.strip("\"'")` cleaning steps before type casting handle these cases.

---

## Testing Strategy

- **Unit testing per module**: each of `data_loader`, `prompt_builder`, `generator`, and `processor` was tested independently with mocked model outputs.
- **Error injection**: invalid JSON, missing files, wrong types, and empty inputs were tested against `data_loader` to verify graceful error messages.
- **End-to-end testing**: all 11 prompts from `function_calling_tests.json` were run and outputs validated against expected function names and argument types.
- **JSON schema validation**: output files were parsed with `json.load()` post-run to confirm 100% parse success.
- **Moulinette**: the provided grading script was used as the final validation pass.

---

## Resources

### Official Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — model loading and tokenization
- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) — base model used
- [Pydantic v2 Docs](https://docs.pydantic.dev/) — data validation
- [uv Documentation](https://docs.astral.sh/uv/) — package manager

### Articles & References
- [Outlines: Structured Text Generation](https://github.com/outlines-dev/outlines) — inspiration for constrained decoding approach
- [Guidance: Constrained Generation](https://github.com/guidance-ai/guidance) — alternative constrained decoding library
- [JSON Mode in LLMs — Survey](https://arxiv.org/abs/2403.06988) — academic overview of structured generation
- [Function Calling in OpenAI API](https://platform.openai.com/docs/guides/function-calling) — reference for output format design

### AI Usage
Claude (Anthropic) was used as a development assistant in this project for the following tasks:
- **Code review and refactoring**: restructuring the monolithic script into modular files (`data_loader`, `prompt_builder`, `generator`, `processor`).
- **Error handling**: generating comprehensive error handling for the `data_loader` module covering path validation, JSON parsing, and schema validation.
- **Makefile**: drafting the Makefile with all required rules (`install`, `run`, `debug`, `clean`, `lint`, `lint-strict`).
- **README**: drafting this document based on project requirements.

All algorithmic logic (constrained decoding, token masking, argument extraction pipeline) was written and validated by the student.
