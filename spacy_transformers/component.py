import spacy
from spacy.language import Language
from spacy.util import registry
from spacy_transformers.wrapper.hf_wrapper import TransformerWrapper


@registry.architectures.register("spacy-transformers.TransformerModel")
def build_transformer_model(name: str) -> TransformerWrapper:
    return TransformerWrapper(name=name)


@Language.factory("transformer_model", assigns=["token._.transformer_data"])
def make_transformer_model(nlp: Language, name: str):
    return TransformerComponent(nlp, name=name)


class TransformerComponent:
    def __init__(self, nlp: Language, name: str):
        self.model = registry.architectures.get("spacy-transformers.TransformerModel")(name=name)

    def __call__(self, doc):
        doc = self.model(doc)
        return doc