import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from embedwise.embedders import SentenceEncoder, HuggingFaceEncoder


sentence_encoder = SentenceEncoder(name="all-MiniLM-L6-v2")
hf_encoder = HuggingFaceEncoder(model_name="bert-base-uncased")

dataset = load_dataset("PolyAI/banking77")
# print(dataset['train']['text'][0])
# print(dataset['test']['text'][0])

text_to_embed = dataset['train']['text'][:100]
# print(f"text_to_embed: \n{text_to_embed}")
# embeddings = encoder.transform(text_to_embed)
# print(f"embedding: \n{embeddings}")
# print(type(embeddings))
# for emb in embeddings:
#     print(f"emb.shape: {emb.shape}\t emb.dtype: {emb.dtype}")



# Generate embeddings
sentence_embeddings = sentence_encoder.get_embedding(text_to_embed)
hf_embeddings = hf_encoder.get_embedding(text_to_embed)

# Compute cosine similarities (example with SentenceEncoder embeddings)
cosine_sim = cosine_similarity(sentence_embeddings)

# Find the most similar pairs (excluding self-comparisons)
np.fill_diagonal(cosine_sim, np.nan)
similar_pair_indices = np.unravel_index(np.nanargmax(cosine_sim), cosine_sim.shape)

most_similar_texts = (text_to_embed[similar_pair_indices[0]], text_to_embed[similar_pair_indices[1]])
similarity_score = cosine_sim[similar_pair_indices]

print("Most similar texts:", most_similar_texts)
print("Similarity score:", similarity_score)

# tsne = TSNE(n_components=2)
# embeddings_2d = tsne.fit_transform(embeddings)


# data = pd.DataFrame(embeddings_2d, columns=["x", "y"])
# data["text"] = text_to_embed

# points = hv.Points(data, kdims=['x', 'y'], vdims=['text']).opts(tools=['box_select', 'lasso_select'])

# # Create Table
# columns = ['x', 'y', 'text']
# table = hv.Table([], columns)

# # Create Selection stream
# selection = Selection1D(source=points)

# # Create DynamicMap
# def selected_info(index):
#     if index:
#         return hv.Table(data.iloc[index], columns)
#     else:
#         return hv.Table([], columns)

# dynamic_table = hv.DynamicMap(selected_info, streams=[selection])

# layout = points + dynamic_table

