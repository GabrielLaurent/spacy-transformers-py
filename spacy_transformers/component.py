import spacy
from spacy.language import Language
from spacy.util import registry
from typing import Optional, Dict, Any

from .wrapper import PyTorchTransformer
from .util import get_model_name, init_transformer_model


@registry.architectures.register("spacy-transformers.TransformerModel")
def TransformerModel(
hf_model_name: str,
model_config: Dict[str, Any] = {},
)
    return init_transformer_model(hf_model_name, model_config)


@Language.factory(
    "transformer",
    assigns=["token.vector"],
    default_config={
        "model_config": {},
        "model": {"@architectures": "spacy-transformers.TransformerModel"},
        "tokenizer_config": {},
    },
    default_score_weights={"transformer_token_vector": 1.0},
)

def make_transformer(
    nlp: Language,
    name: str,
    model,
    model_config: Dict[str, Any],
    tokenizer_config: Dict[str, Any],
):
    return PyTorchTransformer(nlp.vocab, model=model, model_config=model_config, tokenizer_config=tokenizer_config)
