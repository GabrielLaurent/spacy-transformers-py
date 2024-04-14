import pytest
import torch
from spacy_transformers.wrapper.hf_wrapper import HuggingFaceWrapper
from transformers import AutoModel, AutoTokenizer


@pytest.fixture
def hf_wrapper():
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    wrapper = HuggingFaceWrapper(model, tokenizer)
    return wrapper


def test_wrapper_output_shape(hf_wrapper):
    texts = ["This is a test sentence.", "Another test sentence."]
    encoding = hf_wrapper.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    output, _, _ = hf_wrapper(encoding)
    assert output.shape[0] == len(texts)
    assert output.shape[1] == encoding['input_ids'].shape[1]


def test_wrapper_output_type(hf_wrapper):
    texts = ["This is a test sentence.", "Another test sentence."]
    encoding = hf_wrapper.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    output, _, _ = hf_wrapper(encoding)
    assert isinstance(output, torch.Tensor)


def test_wrapper_data_integrity(hf_wrapper):
    texts = ["This is a test sentence.", "Another test sentence."]
    encoding = hf_wrapper.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    output, _, _ = hf_wrapper(encoding)
    assert not torch.isnan(output).any()
