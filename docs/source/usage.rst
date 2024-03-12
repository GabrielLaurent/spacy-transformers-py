Usage
=====

This section provides detailed instructions on how to use the `spacy-transformers` library.

Quick Start
-----------

1.  Install the library:

    ```bash
    pip install spacy-transformers
    ```

2.  Load a spaCy model:

    ```python
    import spacy

    nlp = spacy.load("en_core_web_sm")
    ```

3.  Add the Transformer component to the pipeline:

    ```python
    from spacy_transformers import TransformerData
    from spacy_transformers import TransformerModel

    config = {
        "model_name": "bert-base-uncased",  # Replace with your desired transformer model
        "transformer_data_fn": lambda docs, trf, get_spans:
                TransformerData(docs, trf, get_spans, is_stateful=False),
    }

    # Replace 'transformer' with the name you want to use for the pipeline component.
    nlp.add_pipe("transformer", config=config)
    ```

4.  Process text:

    ```python
    doc = nlp("This is a sample sentence.")
    print(doc._.trf_data.tensors.shape)
    ```


Configuration
-------------

The `Transformer` component can be configured using a dictionary passed to the `nlp.add_pipe` method. The following configuration options are available:

*   `model_name`:  The name of the Hugging Face Transformers model to use.
*   `transformer_data_fn`:  A function that takes a list of spaCy `Doc` objects and a Transformers model, and returns a `TransformerData` object.  This function is responsible for mapping the transformer's output to spaCy `Doc` objects.
*   `get_spans`:  A function to create spans from the transformer output.


Custom Pipelines
---------------

You can create a custom pipeline that includes the `Transformer` component.

```python
import spacy
from spacy.language import Language
from spacy_transformers import TransformerModel

@Language.factory("my_custom_transformer")
def create_custom_transformer(nlp: Language, name: str, model_name: str):
    return TransformerModel.from_pretrained(nlp, model_name)

nlp = spacy.blank("en")
nlp.add_pipe("my_custom_transformer", config={"model_name": "bert-base-uncased"})
doc = nlp("This is a sentence.")
print(doc._.trf_data.tensors.shape)
```