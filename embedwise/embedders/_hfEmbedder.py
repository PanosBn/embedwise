from .baseEmbedder import BaseEmbeddingclass
from transformers import AutoModel, AutoTokenizer

class _HFEmbedder(BaseEmbeddingclass):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _generate_document_embedding(self, text):
        

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return self.to_device(outputs.last_hidden_state)
    