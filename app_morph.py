from transformers import pipeline

oracle = pipeline("ner", model="dicta-il/dictabert-ner", aggregation_strategy="simple")

# if we set aggregation_strategy to simple, we need to define a decoder for the tokenizer. Note that the last wordpiece of a group will still be emitted
from tokenizers.decoders import WordPiece

oracle.tokenizer.backend_tokenizer.decoder = WordPiece()

sentence = """דוד בן-גוריון (16 באוקטובר 1886 - ו' בכסלו תשל"ד) היה מדינאי ישראלי וראש הממשלה הראשון של מדינת ישראל."""
output_dict = oracle(sentence)
print(output_dict)

# from transformers import AutoModel, AutoTokenizer
# import json

# model = AutoModel.from_pretrained("dicta-il/dictabert-morph", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-morph")

# model.eval()

# sentence = """שלום קוראים לי יהויכין בן יהושפט אמדאוס השלישי, ומשנת 1984 אני כל שנה קורא את 1984 של ג'ורג' אורוול. רציתי להציע לכם עוגיות וסטלות טובות ממש. מובטח לכם הנאה צרופה משהו מטורף יעני מילה מבטיח"""
# output_dict = model.predict([sentence], tokenizer)
# print(output_dict)

# Save the dictionary into a JSON file
# with open("output_he_morph.json", "w", encoding="utf-8") as json_file:
#     json.dump(output_dict, json_file, ensure_ascii=False, indent=4)
