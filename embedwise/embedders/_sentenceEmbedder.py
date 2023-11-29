import numpy as np
from .baseEmbedder import BaseEmbedding
from torch.nn import Linear
from torch.quantization import quantize_dynamic
from sentence_transformers import SentenceTransformer as SBERT

class SentenceEncoder(BaseEmbedding):
    """
    Encoder that generate embeddings based on the famous sentence-transformers library.

    For more information, [sentence-transformers docs page](https://www.sbert.net/docs/pretrained_models.html#model-overview).

    Arguments:
        name: name of model, see available options
        device: manually override cpu/gpu device, tries to grab gpu automatically when available
        quantize: turns on quantization
        num_threads: number of treads for pytorch to use, only affects when device=cpu

    """    

    def __init__(self, name="all-MiniLM-L6-v2", quantize=False, num_threads=None):
        super().__init__()
        self.name = name
        self.model = SBERT(name, device=self.device)
        self.num_threads = num_threads
        self.quantize = quantize
        if quantize:
            self.model = quantize_dynamic(self.model, {Linear})
        if num_threads:
            if self.device.type == "cpu":
                self.device.set_num_threads(num_threads)

    def get_embedding(self, text):

        embeddings = self.model.encode(text)
        
        return np.array(embeddings)