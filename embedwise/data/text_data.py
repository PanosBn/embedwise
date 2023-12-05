class TextDataPoint:
    def __init__(self, text, metadata=None, label=None):
        self.text = text
        self.metadata = metadata or {}
        self.label = label or None

    def get_label(self):
        return self.label

    def set_embedding(self, embedding):
        self.embedding = embedding

    def get_embedding(self, model):
        # Implement embedding logic
        self.embedding = model.encode(self.text)