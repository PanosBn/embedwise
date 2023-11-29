import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.manifold import TSNE



from embedwise.embedders import SentenceEncoder


encoder = SentenceEncoder('all-MiniLM-L6-v2')

dataset = load_dataset("PolyAI/banking77")
print(dataset['train']['text'][0])
print(dataset['test']['text'][0])

text_to_embed = dataset['train']['text'][:5000]
print(f"text_to_embed: \n{text_to_embed}")
embeddings = encoder.transform(text_to_embed)
print(f"embedding: \n{embeddings}")
print(type(embeddings))
for emb in embeddings:
    print(f"emb.shape: {emb.shape}\t emb.dtype: {emb.dtype}")


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

