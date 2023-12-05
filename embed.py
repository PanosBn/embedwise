"""
This file contains examples of how to use the embedders.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset

from embedwise.embedders import HuggingFaceEncoder, SentenceEncoder

sentence_encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2")
hf_encoder = HuggingFaceEncoder(model_name="bert-base-uncased")

dataset = load_dataset("PolyAI/banking77")


text_to_embed = dataset["train"]["text"][:40]


hf_embeddings = hf_encoder.get_embedding(text_to_embed)
stf_embeddings = sentence_encoder.get_embedding(text_to_embed)


reduced_hf_embeddings = hf_encoder.reduce_dimensions(hf_embeddings, method="tsne")
reduced_stf_embeddings = sentence_encoder.reduce_dimensions(
    stf_embeddings, method="tsne"
)


print(f"reduced_hf_embeddings: \n{reduced_hf_embeddings}\n\n\n")
print(f"reduced_stf_embeddings: \n{reduced_stf_embeddings}")
