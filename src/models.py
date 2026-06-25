from typing import Any
from pydantic import BaseModel, field_validator


class ParameterInfo(BaseModel):
    """A single function parameter with its type."""
    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid = {"string", "number", "integer", "boolean"}
        if v not in valid:
            raise ValueError(f"Unsupported type '{v}'. Must be one of {valid}")
        return v


class FunctionDefinition(BaseModel):
    """A callable function with its name, description, and parameters."""
    name: str
    description: str
    parameters: dict[str, ParameterInfo]
    returns: ParameterInfo


class Prompt(BaseModel):
    """A single user prompt from the test file."""
    prompt: str


class FunctionCall(BaseModel):
    """The output result for a single processed prompt."""
    prompt: str
    fn_name: str
    args: dict[str, Any]
