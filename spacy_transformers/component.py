import logging

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .data_processing import batch_by_length
from .util import replace_listeners, log_config


logger = logging.getLogger(__name__)


@spacy.component("transformer")
class Transformer:  # Renamed for clarity
    def __init__(self, vocab: spacy.vocab.Vocab, model, name: str = "transformer"):
        self.name = name
        self.model = model
        self.vocab = vocab
        self.logger = logging.getLogger(f"{__name__}.{name}")
        log_config(self.logger, {"model": str(model)})

    def __call__(self, doc: Doc):
        try:
            self.logger.debug(f"Calling transformer on doc: '{doc.text}'")
            encoding = self.model.predict([doc])
            doc._.set("transformer_output", encoding)
            self.logger.debug(f"Transformer output shape: {encoding.shape if encoding is not None else None}")
            return doc
        except Exception as e:
            self.logger.exception(f"Error processing doc: '{doc.text}'")
            raise e


@spacy.language.Language.factory(
    "transformer",
    default_config={
        "model": {
            "@layers": "spacy-transformers.TransformerModel.v1",
        }
    },
)
def make_transformer(
    nlp: Language,
    name: str,
    model,
):
    return Transformer(nlp.vocab, model, name=name)


@Language.component("transformer_pipe")
def transformer_pipe(
doc,
model
):
    try:
        encoding = model.predict([doc])
        doc._.set("transformer_output", encoding)
        return doc
    except Exception as e:
        logger.exception(f"Error processing doc: '{doc.text}'")
        raise e