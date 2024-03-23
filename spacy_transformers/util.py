from typing import Optional, Dict, Any
from transformers import AutoModel

def get_model_name(name: str) -> str:
    '''
    Helper function to extract the model name from a string.
    '''
    if "::" in name:
        name = name.split("::", 1)[1]
    return name


def init_transformer_model(hf_model_name: str, model_config: Dict[str, Any]) -> AutoModel:
    '''
    Initializes the transformer model based on the provided name and config.
    '''
    model_name = get_model_name(hf_model_name)
    try:
        model = AutoModel.from_pretrained(model_name, **model_config)
    except Exception as e:
        print(f"Error initializing model {model_name}: {e}")
        raise

    return model
