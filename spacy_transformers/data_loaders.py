import spacy
import torch
from typing import List, Tuple


def load_data(file_path: str, nlp: spacy.Language) -> List[Tuple[str, str]]:
    """Loads data from a text file where each line contains a text and a label, separated by a tab.

    Args:
        file_path (str): The path to the data file.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the text and the label.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                text, label = line.strip().split('\t')
                data.append((text, label))
            except ValueError:
                print(f"Skipping line: {line.strip()} due to incorrect format")

    return data


def preprocess_data(
    data: List[Tuple[str, str]], nlp: spacy.Language, max_length: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Preprocesses the input text data into token IDs and creates labels.

    Args:
        data (List[Tuple[str, str]]): A list of tuples containing text and labels.
        nlp (spacy.Language): A spaCy language model for tokenization. Must be consistent
            with the tokenizer used in any pre-trained models.
        max_length (int): The maximum sequence length for padding.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor]]: A tuple containing lists of token IDs and labels.
    """
    texts, labels = zip(*data)
    token_ids = []
    processed_labels = []

    for text in texts:
        doc = nlp(text)
        # Convert tokens to IDs using vocabulary.  Assumes tokens exist in vocabulary.
        # A more robust implementation might use <UNK> token for out-of-vocabulary situations
        tokens = [token.text for token in doc]
        ids = [nlp.vocab[token].orth for token in tokens]

        # Pad or truncate sequences to max_length
        ids = ids[:max_length]
        padding_length = max(0, max_length - len(ids))
        ids.extend([0] * padding_length)  # Pad with 0 (or a padding token ID)
        token_ids.append(torch.tensor(ids))

    # Assuming labels are numerical (e.g., 0 or 1).  This should be adapted based on label type
    for label in labels:
        try:
            processed_labels.append(torch.tensor(int(label)))
        except ValueError:
            print(f"Warning: Invalid label '{label}'. Skipping.")
            continue # Skip if the label is not convertible to an integer

    return token_ids, processed_labels


def create_attention_mask(token_ids: List[torch.Tensor]) -> List[torch.Tensor]:
    """Creates an attention mask for the token IDs.

    Args:
        token_ids (List[torch.Tensor]): A list of token ID tensors.

    Returns:
        List[torch.Tensor]: A list of attention mask tensors.
    """
    attention_masks = []
    for ids in token_ids:
        attention_mask = (ids != 0).int()  # 1 for real tokens, 0 for padding tokens
        attention_masks.append(attention_mask)
    return attention_masks