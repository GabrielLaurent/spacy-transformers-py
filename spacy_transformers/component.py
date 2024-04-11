import spacy
from spacy.language import Language
from spacy.tokens import Doc
from typing import Callable, Optional, List


@spacy.registry.architectures.register("spacy-transformers.TransformerModel")
def build_transformer_model(name: str) -> Callable[[List[Doc]], List[List[float]]]:
    """Build a transformer model.

    name (str): The name of the transformer model to use.
    
    RETURNS (Callable[[List[Doc]], List[List[float]]]): A function that takes a list of Doc objects and returns a list of lists of floats.
    """
    def forward(docs: List[Doc]) -> List[List[float]]:
        # Replace this with actual transformer forward pass logic using the model name
        # This is a placeholder, it just returns a list of lists of zeros with the
        # correct dimensions for demonstration purposes
        results = []
        for doc in docs:
            results.append([[0.0] * 768] * len(doc))
        return results
    return forward


@Language.factory("transformer", default_config={
    "name": "bert-base-uncased",  # Example default model
    "model": {"@architectures": "spacy-transformers.TransformerModel", "name": "${name}"}}
)
def make_transformer(nlp: Language, name: str, model: Callable[[List[Doc]], List[List[float]]]):
    """Create a transformer component.

    nlp (Language): The spaCy language object.
    name (str): The name of the component.
    model (Callable[[List[Doc]], List[List[float]]]): A callable that takes a list of Doc objects and returns a list of lists of floats.
    
    RETURNS (Transformer):
    """
    return Transformer(nlp, name, model)


class Transformer:
    """spaCy component that wraps a transformer model.

    model (Callable[[List[Doc]], List[List[float]]]): A callable that takes a list of Doc objects and returns a list of lists of floats.
    """
    def __init__(self, nlp: Language, name: str, model: Callable[[List[Doc]], List[List[float]]]):
        self.nlp = nlp
        self.name = name
        self.model = model

    def __call__(self, doc: Doc) -> Doc:
        """Process a Doc object.

        doc (Doc): The Doc object to process.
        
        RETURNS (Doc):
        """
        # Get the transformer output for the document
        transformer_output = self.model([doc])

        # Add the transformer output to the Doc object
        doc.user_data["transformer_output"] = transformer_output

        return doc