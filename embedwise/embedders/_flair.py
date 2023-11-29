import logging
from flair.data import Sentence
from .baseEmbedder import BaseEmbedding
from flair.embeddings import TransformerDocumentEmbeddings
from typing import List

class FlairEncoder(BaseEmbedding):
    """
    A class that represents a Flair document encoder for text embedding.

    Attributes:
        model_name (str): The name of the pre-trained model to be used for encoding.

    Methods:
        get_embedding(text): Returns the embedding representation of the given text.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model = TransformerDocumentEmbeddings(model_name)
        self.logger.info(f"Loaded Flair model {model_name}")

    def get_embedding(self, text: str | List[str]):
        if isinstance(text, str):
            return self._embed_text(text)
        elif isinstance(text, list):
            return [self._embed_text(t) for t in text]
        else:
            self.logger.error("Input must be a string or a list of strings.")
            return None

    def _embed_text(self, text: str):
        sentence = Sentence(text)
        self.model.embed(sentence)
        
        return sentence.get_embedding().detach().cpu().numpy()
