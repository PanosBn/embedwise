import torch
from .baseEmbedder import BaseEmbedding
from transformers import AutoModel, AutoTokenizer

class HuggingFaceEncoder(BaseEmbedding):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def get_embedding(self, text):
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
            print(e)
            return None
            
