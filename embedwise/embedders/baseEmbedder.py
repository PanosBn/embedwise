import embedwise
import logging
import numpy
from numpy import ndarray
from typing import List

logging.basicConfig(level=logging.DEBUG, filename="embedders.log", filemode="w")

class BaseEmbedding:
    def __init__(self):
        self.device = embedwise.device
        self.logger = logging.getLogger(__name__)


    def get_embedding(self, input):
        raise NotImplementedError("This method should be overridden by subclass")
    
    def reduce_dimensions(self, embeddings: numpy.ndarray, n_components: int = 2, method='tsne'):
        """
        Reduce the dimensionality of the given embeddings using the given method.

        Args:
            embeddings (numpy.ndarray): The embeddings to be reduced.
            n_components (int): The number of components to reduce to.
            method (str): The method to use for dimensionality reduction. 
                Currently only 'tsne' is supported.

        Returns:
            numpy.ndarray: The reduced embeddings.

        """
        if method == 'tsne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=n_components)
            embeddings_2d = tsne.fit_transform(embeddings)
            return embeddings_2d
        else:
            self.logger.error(f"Method {method} not supported.")
            return None
