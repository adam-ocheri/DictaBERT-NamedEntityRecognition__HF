from transformers import pipeline
from shared_data import cache_dir, model_name


oracle = pipeline("ner", model=model_name, aggregation_strategy="simple")

# if we set aggregation_strategy to simple, we need to define a decoder for the tokenizer. Note that the last wordpiece of a group will still be emitted
from tokenizers.decoders import WordPiece

oracle.tokenizer.backend_tokenizer.decoder = WordPiece()

oracle.model.save_pretrained(cache_dir + model_name + "/model")
oracle.tokenizer.save_pretrained(cache_dir + model_name + "/tokenizer")
