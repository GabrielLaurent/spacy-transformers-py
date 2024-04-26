# spacy_transformers/wrapper.py

from transformers import AutoModel, AutoTokenizer

class TransformerWrapper:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, text: str):
        # Tokenize the input text and return
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        #adapt transformer outputs and make them compatible with spaCy's Doc objects

        return {"token_embeddings": outputs.last_hidden_state.tolist()[0]} #Return a list of lists
