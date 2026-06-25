*This project has been created as part of the 42 curriculum by mbougajd.*

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

**Call Me Maybe** is a Python project that demonstrates how a local Large Language Model (LLM) can perform function calling without relying on external APIs.

The program receives a natural language request and converts it into a structured JSON function call matching one of the available function definitions.

For example:

**Input**

```text
What is the sum of 2 and 3?
```

**Output**

```json
{
  "fn_name": "fn_add_numbers",
  "args": {
    "a": 2,
    "b": 3
  }
}
```

The project uses **constrained decoding** to ensure that every generated output follows both the required JSON structure and the function schema.

---

## Goal

The objective of this project is to bridge natural language understanding and structured function execution.

Instead of generating free-form text, the model must:

1. Select the correct function from a predefined list.
2. Extract the required arguments from the user's request.
3. Generate a valid JSON object matching the expected schema.
4. Prevent invalid outputs through constrained decoding.

---

## Features

* Local LLM inference
* Function selection from predefined definitions
* Schema-aware constrained decoding
* JSON output generation
* Pydantic-based validation
* Deterministic generation
* No external function-calling APIs

---

## Project Structure

```text
call-me-maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py
│   ├── models.py
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
├── uv.lock
└── README.md
```

---

## Algorithm Explanation

### Overview

The implementation relies on **constrained decoding**, a generation technique that restricts the set of tokens the model is allowed to produce at each step.

Unlike traditional text generation, where the model may generate any token from its vocabulary, constrained decoding filters invalid tokens before selecting the next one.

This guarantees that the generated output remains valid throughout the entire generation process.

### Step 1: Function Selection

The first stage consists of selecting the appropriate function.

1. All available function names are loaded from the function definition file.
2. Function names are tokenized in advance.
3. During generation, only tokens that can continue a valid function name remain available.
4. All other token probabilities are masked.
5. Generation stops when a complete valid function name has been produced.

For example, if the valid functions are:

```text
fn_add_numbers
fn_subtract_numbers
fn_get_square_root
```

and generation starts with:

```text
fn_
```

only tokens that can continue one of the valid names are allowed.

### Step 2: Argument Extraction

After selecting the function, the corresponding schema is loaded.

For each parameter:

1. The model receives the user prompt and function context.
2. Allowed tokens are restricted according to the parameter type.
3. The generated value is validated before being accepted.

Examples:

* Integer parameters only accept valid integer values.
* Float parameters only accept valid floating-point values.
* String parameters must produce valid JSON strings.
* Required parameters must always be generated.

### Step 3: JSON Assembly

Once all parameters have been generated:

1. The function name and arguments are assembled.
2. The result is converted into a JSON object.
3. The JSON is validated against the schema.

Example:

```json
{
  "fn_name": "fn_get_square_root",
  "args": {
    "a": 16
  }
}
```

This process ensures both:

* Syntactic validity (valid JSON)
* Semantic validity (schema-compliant output)

---

## Design Decisions

### Constrained Decoding

Constrained decoding was chosen because it guarantees structured outputs and prevents invalid generations.

Without constraints, the model could:

* Generate unknown function names
* Produce malformed JSON
* Return invalid argument types

### Two-Stage Generation

Function selection and argument generation are handled separately.

Advantages:

* Simpler implementation
* Easier debugging
* Better control over generation constraints

### Pydantic Validation

Pydantic is used to:

* Validate loaded schemas
* Validate generated outputs
* Provide clear error messages

### Local Inference

The project uses a local model instead of external APIs to ensure reproducibility and independence from third-party services.

---

## Instructions

### Requirements

* Python 3.10+
* uv

### Installation

```bash
git clone <repository-url>
cd call-me-maybe
uv sync
```

### Run

Default execution:

```bash
uv run python -m src
```

Custom input and output files:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

---

## Example Usage

### Example 1

Input:

```text
What is the sum of 2 and 3?
```

Output:

```json
{
  "fn_name": "fn_add_numbers",
  "args": {
    "a": 2,
    "b": 3
  }
}
```

### Example 2

Input:

```text
What is the square root of 16?
```

Output:

```json
{
  "fn_name": "fn_get_square_root",
  "args": {
    "a": 16
  }
}
```

---

## Performance Analysis

### Accuracy

The constrained decoding strategy prevents the generation of unknown function names and significantly reduces invalid outputs.

Generated results are validated against the provided schema before being accepted.

### Speed

The solution remains efficient because invalid tokens are filtered early in the generation process.

Although masking introduces additional computation, the overhead is minimal compared to the cost of model inference.

### Reliability

The implementation consistently produces:

* Valid JSON structures
* Valid function names
* Correctly typed arguments

This makes the system more reliable than unrestricted text generation approaches.

---

## Challenges Faced

### Function Name Tokenization

Different function names may be split into multiple tokens by the tokenizer.

**Solution:**

All valid function names were pre-tokenized and generation was restricted to valid token continuations.

### Generation Termination

Determining when generation should stop can be difficult.

**Solution:**

Explicit stopping conditions were added for completed function names and completed parameter values.

### Type Enforcement

The model may generate values with incorrect types.

**Solution:**

Parameter-specific constraints and validation rules were applied during generation.

### Schema Validation

Generated outputs must exactly match the required schema.

**Solution:**

Pydantic models were used to validate outputs before writing them to the final JSON file.

---

## Testing Strategy

The implementation was validated using several approaches.

### Unit Testing

Individual components were tested independently:

* Function loading
* Schema validation
* Token masking
* Argument extraction

### Integration Testing

The entire pipeline was tested from:

```text
User Prompt
    ↓
Function Selection
    ↓
Argument Generation
    ↓
JSON Validation
```

### Dataset Testing

The provided test dataset was executed multiple times to verify deterministic and consistent outputs.

### Output Validation

Every generated result was checked to ensure:

* Valid JSON syntax
* Existing function names
* Correct parameter types
* Schema compliance

### Moulinette Evaluation

The final implementation was validated using the official Moulinette evaluation process.

---

## Resources

### Documentation

* Hugging Face Transformers Documentation
* Pydantic Documentation
* Python Documentation
* JSON Schema Documentation
* uv Documentation

### Articles and Tutorials

* Function Calling with LLMs
* Structured Generation Techniques
* Constrained Decoding for Language Models
* Tokenization and Text Generation

### AI Usage

AI tools were used during the development of this project for:

* Understanding constrained decoding concepts
* Reviewing implementation ideas
* Debugging Python code
* Improving documentation quality
* Generating and refining test cases
* Refactoring and code quality suggestions

All design decisions, implementation details, and final code were reviewed and validated manually.
