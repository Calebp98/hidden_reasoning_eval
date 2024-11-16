from inspect_ai.dataset import FieldSpec, json_dataset

dataset1 = json_dataset("reliability_results_cot_critical.json",FieldSpec(input="question", target="answer", metadata=["raw_answer"]))

print(dataset1)
