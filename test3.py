def get_param_prompt(self, func_def: str,
                         function_name: str, param_name: str,
                         param_type: str, prompt: str,
                         parameters: dict[str, Any],) -> str:
        params_context = ""
        if parameters:
            for key, value in parameters.items():
                params_context += (
                    f"\"{key}\": {json.dumps(value)},\n            "
                )

        if param_type in {"int", "integer", "float", "number"}:
            value_prefix = ""
        elif param_type in {"bool", "boolean"}:
            value_prefix = ""
        else:
            value_prefix = '"'

        return f"""Available functions:
{func_def}

User prompt: {prompt}

Extract the '{param_name}' parameter for the function \
'{function_name}' and complete the JSON:
{{
        \"prompt\": {json.dumps(prompt)},
        \"name\": \"{function_name}\",
        \"parameters\": {{
            {params_context}\"{param_name}\": {value_prefix}"""