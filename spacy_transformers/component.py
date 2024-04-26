# spacy_transformers/component.py

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from .wrapper import TransformerWrapper

@Language.factory(
    "transformer_component",
    default_config={"model_name": "bert-base-uncased"},
)
class TransformerComponent:
    def __init__(self, nlp: Language, name: str, model_name: str):
        self.nlp = nlp
        self.name = name
        self.wrapper = TransformerWrapper(model_name)

    def __call__(self, doc: Doc):
        # This is where the transformer is applied to the doc
        # For a basic example, let's add an attribute with token embeddings
        transformer_output = self.wrapper(doc.text)

        # Assuming wrapper.forward returns a dictionary with 'token_embeddings'
        doc.set_extension("transformer_output", default=transformer_output, force=True) # save detailed information to custom attribute

        if 'token_embeddings' in transformer_output:
            for i, token in enumerate(doc):
                token.set_extension("embedding", default=transformer_output['token_embeddings'][i], force=True)
        return doc