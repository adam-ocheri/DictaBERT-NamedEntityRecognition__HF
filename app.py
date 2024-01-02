import json
from transformers import pipeline
from pipeline_actions import run_pipeline
from shared_data import NpEncoder

example_sentence = """דוד בן-גוריון (16 באוקטובר 1886 - ו' בכסלו תשל"ד) היה מדינאי ישראלי וראש הממשלה הראשון של מדינת ישראל."""


def main(sentence):
    model, tokenizer = run_pipeline()
    oracle = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )

    output_dict = oracle(sentence)
    print(output_dict)

    with open("result.json", "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, cls=NpEncoder, ensure_ascii=False, indent=4)


main(example_sentence)
