# EmbedWise: Comparative Analysis of Text Embeddings

## Overview
EmbedWise is a toolkit enabling the comparison of deep learning models from various sources (e.g., Hugging Face, Flair, SentenceTransformers) in a single task. This tool is invaluable for determining which model best fits specific Natural Language Processing (NLP) tasks. With the rapid advancement in NLP, the ability to efficiently evaluate and compare models is crucial. EmbedWise facilitates this by abstracting the process of model loading, allowing the generation, visualization, and analysis of text embeddings from diverse sources.

## Key Features
- **Model Agnosticism**: Easy integration with popular NLP libraries.
- **Embedding Generation**: Efficient creation of sentence or paragraph embeddings from multiple pre-trained models.
- **Dimensionality Reduction**: Built-in support for techniques like PCA and t-SNE to visualize high-dimensional data.
- **Comparative Visualization**: Tools for qualitative analysis through embedding plotting and comparison.
- **User-Friendly**: Simple and intuitive API, accessible for both beginners and advanced users.

## Installation

Ensure Python 3.10 or later is installed, along with Poetry. Spawning a virtual environment within poetry is recommended.

```bash
# Clone the repository
git clone https://github.com/PanosBn/EmbedWise.git
cd EmbedWise

# Install with poetry
poetry install
```

```python
import pandas as pd
import altair as alt
from embedwise import SentenceEncoder, HuggingFaceEncoder, FlairDocumentEncoder


# Initialize encoders
hf_encoder = HuggingFaceEncoder("your-hf-model-name")
sent_encoder = SentenceEncoder("your-sentence-model-name")

# Example texts
texts = ["Sample text for encoding", "Another sample text"]

# Generate embeddings
hf_embeddings = hf_encoder.get_embedding(texts)
sent_embeddings = sent_encoder.get_embedding(texts)

# Reduce dimensions for visualization
reduced_hf_embeddings = hf_encoder.reduce_dimensions(hf_embeddings, method='tsne')
reduced_sent_embeddings = sent_encoder.reduce_dimensions(sent_embeddings, method='tsne')

```
## Vizualizing the embeddings with Altair
```python
# Prepare DataFrame
df = pd.DataFrame({
    'x': reduced_hf_embeddings[:, 0],
    'y': reduced_hf_embeddings[:, 1],
    'label': ['Label1', 'Label2']
})

# Create Altair scatter plot
chart = alt.Chart(df).mark_circle(size=60).encode(
    x='x', y='y', color='label:N', tooltip=['label:N']
).interactive()

chart.show()
```

