from llm_sdk.llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

decode = list(model.encode("hello"))
result = model.get_logits_from_input_ids(decode)
res_ktba = model.decode(result)

print(res_ktba)