# tests/conftest.py

import pytest
import spacy
from spacy.language import Language

@pytest.fixture(scope="session")
def nlp() -> Language:
    return spacy.blank("en")