import embedwise

class BaseEmbedding:
    def __init__(self):
        self.device = embedwise.device

    def get_embedding(self, input):
        raise NotImplementedError("This method should be overridden by subclass")