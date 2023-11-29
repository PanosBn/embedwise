import torch
from .baseEmbedder import BaseEmbedding
from transformers import AutoModel, AutoTokenizer
from typing import List


class HuggingFaceEncoder(BaseEmbedding):
    """
    A class that represents a Hugging Face encoder for text embedding.

    Attributes:
        model_name (str): The name of the pre-trained model to be used for encoding.

    Methods:
        get_embedding(text): Returns the embedding representation of the given text.

    """

    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.logger.info(f"Loaded model {model_name} on device {self.device}")


    def get_embedding(self, text: str | List[str]):
        """
        Get the embedding for the given text or list of texts.

        Args:
            text (str or List[str]): The input text or list of texts to get embeddings for.

        Returns:
            numpy.ndarray: The embeddings of the input text(s) as a numpy array.

        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            # @DECISION - If the model return token embeddings, average them
            # If the model returns sentence embeddings use them directly
            if outputs.last_hidden_state.size(1) > 1:  
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:  
                embeddings = outputs.last_hidden_state.squeeze(1)

            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            return embeddings.cpu().numpy()
        except Exception as e:
            self.logger.error(f"Error in get_embedding: {e}")
            return None
        
