from llm_sdk.llm_sdk import Small_LLM_Model
import json

def build_prompt(user_question: str, functions: list) -> str:
    functions_list = ""
    for fn in functions:
        args = ", ".join(
            f'"{param_name}": {param_info["type"]}'
            for param_name, param_info in fn["parameters"].items()
        )
        functions_list += f'- {fn["name"]}: {{{args}}}\n'

    return f"""You are a strict function-calling assistant.
Available functions:
{functions_list}
User question: {user_question}

Respond ONLY with valid JSON."""


class LazyJSONDecoder:
    def __init__(self, model: Small_LLM_Model, functions: list):
        self.model = model
        self.function_names = [fn["name"] for fn in functions]
        self.vocab_cache = {}
        self.eos_tokens = ["</s>", "<|end|>", "<|im_end|>", "<|endoftext|>"]
        
    def _is_valid_continuation(self, generated_text: str, tstr: str) -> bool:
            PART1 = '{"name": "'
            PART2 = '", "arguments": {'
            
            proposed = generated_text + tstr
            
            if len(generated_text) < len(PART1):
                return PART1.startswith(proposed) or proposed.startswith(PART1)
                
            after_part1 = generated_text[len(PART1):]
            if '"' not in after_part1:
                proposed_after = after_part1 + tstr
                for fn in self.function_names:
                    if fn.startswith(proposed_after) or proposed_after.startswith(fn + '"'):
                        return True
                return False
                
            name_end_idx = after_part1.index('"')
            after_name = after_part1[name_end_idx:]
            
            if len(after_name) < len(PART2):
                proposed_after = after_name + tstr
                return PART2.startswith(proposed_after) or proposed_after.startswith(PART2)
                
            after_part2 = after_name[len(PART2):]
            
            if self._is_closed(after_part2):
                return tstr in self.eos_tokens

            if tstr in self.eos_tokens or "```" in tstr or "\n" in tstr:
                return False

            proposed_after = after_part2 + tstr
            
            inner_brackets = 1  
            in_string = False
            escape = False
            
            for i, char in enumerate(proposed_after):
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{': 
                        inner_brackets += 1
                    elif char == '}': 
                        inner_brackets -= 1
                        
                        if inner_brackets == 0:
                            remainder = proposed_after[i+1:].strip()
                            
                            if remainder == "":
                                return True
                            elif remainder == "}":
                                return True
                            else:
                                return False
            
            return inner_brackets < 0

    def _is_closed(self, args_string: str) -> bool:
        """Helper to check if BOTH dictionaries are successfully closed."""
        inner_brackets = 1
        in_string = False
        escape = False
        for i, char in enumerate(args_string):
            if escape: escape = False; continue
            if char == '\\': escape = True; continue
            if char == '"': in_string = not in_string; continue
            if not in_string:
                if char == '{': inner_brackets += 1
                elif char == '}': 
                    inner_brackets -= 1
                    if inner_brackets == 0:
                        remainder = args_string[i+1:].strip()
                        return remainder == "}"
        return False

    def get_valid_next_token(self, logits: list, generated_text: str) -> int:
        indexed_logits = list(enumerate(logits))
        indexed_logits.sort(key=lambda x: x[1], reverse=True)
        
        for token_id, _ in indexed_logits:
            if token_id not in self.vocab_cache:
                try:
                    self.vocab_cache[token_id] = self.model.decode(token_id)
                except Exception:
                    self.vocab_cache[token_id] = ""
                    
            tstr = self.vocab_cache[token_id]
            if not tstr:
                continue
                
            if self._is_valid_continuation(generated_text, tstr):
                return token_id
                
        return None

def main():
    Amadeus = Small_LLM_Model()

    with open("data/input/functions_definition.json", "r", encoding="utf-8") as f:
        functions = json.load(f)

    with open("data/input/function_calling_tests.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    decoder = LazyJSONDecoder(model=Amadeus, functions=functions)

    for user_question in prompts:
        prompt = build_prompt(user_question.get("prompt"), functions)
        input_ids = Amadeus.encode(prompt).tolist()[0]

        generated_tokens = []
        generated_text = ""
        
        for _ in range(100):
            logits = Amadeus.get_logits_from_input_ids(input_ids)
            
            next_token = decoder.get_valid_next_token(logits, generated_text)
            
            if next_token is None:
                break
            
            input_ids.append(next_token)
            generated_tokens.append(next_token)
            
            token_text = decoder.vocab_cache[next_token]
            generated_text += token_text
            
            if token_text in decoder.eos_tokens:
                break

        clean_text = generated_text
        for eos in decoder.eos_tokens:
            clean_text = clean_text.replace(eos, "")

        print(f"Q: {user_question.get('prompt')}")
        print(f"A: {clean_text.strip()}")
        print("-" * 40)

if __name__ == "__main__":
    main()