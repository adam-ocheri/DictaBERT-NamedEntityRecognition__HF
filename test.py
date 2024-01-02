from transformers import BertForTokenClassification, AutoTokenizer
import torch
from torch.nn.functional import softmax

# Instantiate the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-ner")
model = BertForTokenClassification.from_pretrained("dicta-il/dictabert-ner")

# Sentence to process
sentence = """דוד בן-גוריון (16 באוקטובר 1886 - ו' בכסלו תשל"ד) היה מדינאי ישראלי וראש הממשלה הראשון של מדינת ישראל."""

# Encode the sentence and convert it into tensors
inputs = tokenizer.encode(sentence, return_tensors="pt")

# Pass the processed input to the model
outputs = model(inputs)

# Extract the logits
logits = outputs.logits

# Apply softmax to get probabilities
probs = softmax(logits, dim=-1)

# Decode the results
results = tokenizer.batch_decode(torch.argmax(probs, dim=-1))

print(results)
