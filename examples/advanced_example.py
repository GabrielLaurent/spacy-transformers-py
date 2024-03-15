import spacy
from spacy_transformers import TransformerData
from spacy.training import Example
import torch

# Custom data loading function
def load_data(limit=0):
    texts = [
        "This is the first sentence.",
        "Here is another, quite similar.",
        "And a third one for good measure.",
    ]
    labels = [{"entities": [(0, 4, "Type1")]}, {"entities": [(0, 4, "Type2")]}, {"entities": [(0, 5, "Type1")]}]

    data = []
    for i, (text, annotation) in enumerate(zip(texts, labels)):
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"entities": annotation['entities']})
        data.append(example)
        if limit > 0 and i >= limit:
            break
    return data

# Load a blank spaCy model
nlp = spacy.blank("en")

# Add the transformer component to the pipeline
transformer = nlp.add_pipe("transformer", config={"model_name": "roberta-base"})
nlp.add_pipe('ner')

# Initialize the weights - VERY IMPORTANT
training_data = load_data(limit=3)
nlp.initialize(lambda: training_data)

# Train the model (minimal example)
optimizer = nlp.resume_training()
for i in range(5):
    losses = {}
    for example in training_data:
        nlp.update([example], sgd=optimizer, losses=losses)
    print(f"Losses: {losses}")

# Sample text
text = "This is a test sentence for Type1."

# Process the text
doc = nlp(text)

# Access the transformer data
transformer_data: TransformerData = doc._.get("transformer_data")

# Print some information
print(f"Text: {text}")
print(f"Number of tokens: {len(doc)}")
print(f"Transformer model name: {transformer.model.transformer.name}")
elapsed = 0.0
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

# Example: Accessing token embeddings
if transformer_data and transformer_data.wordpieces:
    token_embeddings = transformer_data.wordpieces.features
    print(f"Shape of token embeddings: {token_embeddings.shape}")

# Example: Accessing alignment information
if transformer_data and transformer_data.align:
    alignments = transformer_data.align
    print(f"Alignments: {alignments}")