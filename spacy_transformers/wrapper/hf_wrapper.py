from transformers import AutoTokenizer, AutoModel
import spacy
import torch

class HuggingFaceWrapper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, doc: spacy.tokens.Doc) -> torch.Tensor:
        """Tokenizes the text and returns dummy embeddings."""
        inputs = self.tokenizer(doc.text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Return the embeddings (last_hidden_state is a common choice)
        return outputs.last_hidden_state

    def tokenize(self, text: str) -> list:
        """Tokenizes the text using the Hugging Face tokenizer."""
        return self.tokenizer.tokenize(text)