import pytest
import spacy
from spacy_transformers import TransformerData


@pytest.fixture
def nlp():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    return nlp


@pytest.mark.slow
@pytest.mark.parametrize("name", ["bert-base-uncased"])
def test_simple_pipeline(name, nlp):
    # This is a regression test for basic end-to-end functionality.
    try:
        nlp.add_pipe("transformer", config={"name": name}, last=True)
    except OSError:
        pytest.skip(f"{name} not available. Check internet connection.")

    doc = nlp("This is a test sentence.")
    assert doc.has_annotation("ENT_IOB")
    assert len(doc) > 0
    for token in doc:
        assert token.has_vector



@pytest.mark.slow
def test_token_alignment(nlp):
    # Regression test for token alignment.
    text = "This is a sentence ."
    doc = nlp(text)
    words = [t.text for t in doc]
    try:
        nlp.add_pipe("transformer", config={"name": "bert-base-uncased"}, last=True)
    except OSError:
        pytest.skip("bert-base-uncased not available. Check internet connection.")

    doc = nlp(text)
    trf_data: TransformerData = doc._.trf_data

    for i, word in enumerate(words):
        wp_index = trf_data.align[i].dataXd
        # Check alignment is not empty
        assert len(wp_index) > 0

    #Check number of transformer output vectors are correct
    assert len(trf_data.all_outputs['last_layer_only']) == len(doc._.trf_data.align) > 0
