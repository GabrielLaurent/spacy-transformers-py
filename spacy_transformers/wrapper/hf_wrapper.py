import logging
from typing import List

from spacy.tokens import Doc
import torch
from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)

class HFTransformerWrapper:
    def __init__(self, name: str):
        self.name = name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModel.from_pretrained(name)
            self.model.eval()
            self.logger = logging.getLogger(f"{__name__}.{name}")
            self.logger.info(f"Loaded model: {name}")
        except Exception as e:
            self.logger.exception(f"Failed to load model: {name}")
            raise e

    def __call__(self, docs: List[Doc]):
        try:
            self.logger.debug(f"Processing {len(docs)} docs")
            texts = [doc.text for doc in docs]
            encoded_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
            self.logger.debug(f"Transformer output shape: {outputs.last_hidden_state.shape}")
            return outputs.last_hidden_state

        except Exception as e:
            self.logger.exception(f"Error processing docs: {texts[:5]}..." if len(docs) > 0 else "Error processing docs: No documents given" )
            raise e

    def predict(self, docs: List[Doc]):
        return self.__call__(docs)