# examples/example.py

import spacy

# Load a spaCy model (you might need to download one using: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Add the transformer component to the pipeline (replace with your actual component name)
nlp.add_pipe("transformer_component", config={"model_name": "bert-base-uncased"})

# Process some text
doc = nlp("This is a sample sentence.")

# Access the transformer output (assuming it's stored in doc._.transformer_output)
if doc._.has("transformer_output"):
    print(doc._.transformer_output)

    # Access the token embedding (if set by component)
    for token in doc:
        if token._.has("embedding"):
            print(f"Token: {token.text}, Embedding: {token._.embedding[:5]}...") #Print first 5 elements only
