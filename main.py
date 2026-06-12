from llm_sdk.llm_sdk import Small_LLM_Model

Amadeus = Small_LLM_Model()


functions = [
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  },
  {
    "name": "fn_get_square_root",
    "description": "Calculate the square root of a number.",
    "parameters": {
      "a": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_substitute_string_with_regex",
    "description": "Replace all occurrences matching a regex pattern in a string.",
    "parameters": {
      "source_string": {
        "type": "string"
      },
      "regex": {
        "type": "string"
      },
      "replacement": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]
# input_ids = model.encode(
#     f'In this list of functions {functions}, which function is best suited to answer the question: "What is the sum of -2222222 and +333333333333?"'
# ).tolist()[0]

input_ids = Amadeus.encode(
    "In this text, 'What is the sum of 2 and 3?', what is the best word that can be used as an argument in any function?"
).tolist()[0]

print("Input IDs:", input_ids)

while True:


  logits = Amadeus.get_logits_from_input_ids(input_ids)

  big_logit = logits.index(max(logits))

  decod = Amadeus.decode(big_logit)
  print(decod)

  input_ids.append(big_logit)