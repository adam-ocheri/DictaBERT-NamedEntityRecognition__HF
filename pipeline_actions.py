from transformers import pipeline
from shared_data import cache_dir, model_name
from pathlib import Path
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_dir = cache_dir + model_name + "/model"
tokenizer_dir = cache_dir + model_name + "/tokenizer"


def create_pipeline() -> None:
    # TODO: Research additional aggregation-strategies (and learn wtf do they even do)
    oracle = pipeline("ner", model=model_name, aggregation_strategy="simple")

    # if we set aggregation_strategy to simple, we need to define a decoder for the tokenizer. Note that the last wordpiece of a group will still be emitted
    from tokenizers.decoders import WordPiece

    oracle.tokenizer.backend_tokenizer.decoder = WordPiece()

    oracle.model.save_pretrained(model_dir)
    oracle.tokenizer.save_pretrained(tokenizer_dir)


def does_pipeline_exists() -> bool:
    model_path = Path(model_dir)
    tokenizer_path = Path(tokenizer_dir)
    return model_path.exists() and tokenizer_path.exists()


def conditional_pipeline_init() -> None:
    if not does_pipeline_exists():
        create_pipeline()


def run_pipeline():
    conditional_pipeline_init()
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    return model, tokenizer
