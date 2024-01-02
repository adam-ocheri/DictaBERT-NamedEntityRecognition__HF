# from transformers import pipeline

# oracle = pipeline("ner", model="dicta-il/dictabert-ner", aggregation_strategy="simple")

# # if we set aggregation_strategy to simple, we need to define a decoder for the tokenizer. Note that the last wordpiece of a group will still be emitted
# from tokenizers.decoders import WordPiece

# oracle.tokenizer.backend_tokenizer.decoder = WordPiece()

# sentence = """דוד בן-גוריון (16 באוקטובר 1886 - ו' בכסלו תשל"ד) היה מדינאי ישראלי וראש הממשלה הראשון של מדינת ישראל."""
# result = oracle(sentence)

# print("FINISHED PROCESS:")
# print(result)

from transformers import BertForTokenClassification, AutoTokenizer
import torch
import json

# Instantiate the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-ner")
model = BertForTokenClassification.from_pretrained("dicta-il/dictabert-ner")

# Sentence to process
sentence = """דוד בן-גוריון (16 באוקטובר 1886 - ו' בכסלו תשל"ד) היה מדינאי ישראלי וראש הממשלה הראשון של מדינת ישראל."""
# sentence = """David Ben Gurion (October 16th 1886 - October 7 1953) was an Israeli statesman and the first prime minister of the state of Israel."""

# Encode the sentence and convert it into tensors
inputs = tokenizer.encode(sentence, return_tensors="pt")

# Pass the processed input to the model
outputs = model(inputs)

# Extract the output logits
logits = outputs.logits

# Apply softmax to get probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)

# Get the predicted label indices
predicted_indices = torch.argmax(probs, dim=-1)

# Convert the predicted indices to labels
predicted_labels = [model.config.id2label[idx] for idx in predicted_indices.tolist()[0]]

print(predicted_labels)

# # Tokenize the sentence without converting it to tensors
# tokens = tokenizer.tokenize(sentence)

# # Combine tokens and their corresponding predicted labels into a dictionary
# output_dict = dict(zip(tokens, predicted_labels))

# print(output_dict)


# # Save the dictionary into a JSON file
# with open("output_he.json", "w", encoding="utf-8") as json_file:
#     json.dump(output_dict, json_file, ensure_ascii=False, indent=4)
