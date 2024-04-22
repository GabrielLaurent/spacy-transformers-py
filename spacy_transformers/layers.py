from typing import List, Tuple, Dict, Any
import torch
from spacy.tokens import Doc


class TransformerModelOutput:
    def __init__(self, all_layer_representations: List[torch.Tensor], last_layer_only: torch.Tensor):
        self.all_layer_representations = all_layer_representations
        self.last_layer_only = last_layer_only


class TransformerWrapper:
    def __call__(self, docs: List[Doc]) -> Tuple[List[TransformerModelOutput], List[Doc]]:
        raise NotImplementedError

    def from_config(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError

    def to_config(self) -> Dict[str, Any]:
        raise NotImplementedError
