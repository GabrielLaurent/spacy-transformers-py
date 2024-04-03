import pytest
import spacy
from spacy.language import Language


@pytest.fixture
def nlp() -> Language:
    '''Basic spaCy pipeline with the transformer component'''
    nlp = spacy.blank("en")
    # This is a placeholder; replace with your actual component registration
    # and config initialization
    config = {
        "model_name": "bert-base-uncased",  # Or a smaller, faster model
    }

    # This assumes your component is registered as 'transformer'
    # and can be added via nlp.add_pipe("transformer", config=config)
    try:
        nlp.add_pipe("transformer", config=config)
    except ValueError:
        pytest.skip("Could not add transformer pipe. Ensure spacy-transformers is installed and configured correctly.")

    return nlp



def test_component_integration(nlp):
    '''Test that the component integrates correctly into a spaCy pipeline'''
    doc = nlp("This is a test sentence.")

    # Basic check to see if the component has added vectors to the tokens
    # Or has modified the Doc object in some way.
    assert doc.has_annotation("tensor") # Example assertion; adapt based on actual behaviour
    for token in doc:
        assert token.vector.shape[0] > 0



def test_component_with_multiple_docs(nlp):
    '''Test that the component works with multiple documents sequentially'''
    texts = ["This is the first document.", "This is the second document."]
    for text in texts:
        doc = nlp(text)
        assert doc.has_annotation("tensor")
        for token in doc:
            assert token.vector.shape[0] > 0



def test_component_error_handling(nlp):
    '''Test that the component handles errors gracefully'''
    # Example test to check for errors when processing invalid input
    # (Adapt as necessary for your component's specific behaviour)
    with pytest.raises(Exception):
        nlp(None) # or some invalid input