import spacy
from spacy_transformers import TransformerData,


# Load a blank spaCy model
nlp = spacy.blank("en")

# Add the transformer component to the pipeline
# Replace 'roberta-base' with the desired transformer model
transformer = nlp.add_pipe("transformer", config={"model_name": "roberta-base"})

# Sample text
text = "This is a test sentence. Another sentence here."

# Process the text
doc = nlp(text)

# Access the transformer data
transformer_data: TransformerData = doc._.get("transformer_data")

# Print some information
print(f"Text: {text}")
print(f"Number of tokens: {len(doc)}")
print(f"Transformer model name: {transformer.model.transformer.name}")

# Example: Accessing token embeddings
if transformer_data and transformer_data.wordpieces:
    token_embeddings = transformer_data.wordpieces.features
    print(f"Shape of token embeddings: {token_embeddings.shape}")

# Example: Accessing alignment information
if transformer_data and transformer_data.align:
    alignments = transformer_data.align
    print(f"Alignments: {alignments}")