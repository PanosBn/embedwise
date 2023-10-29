from embedwise.utils import NotInstalledError
from ._sentenceEmbedder import SentenceEncoder

# try:
#     from embedwise.embedders._sentenceEmbedder import SentenceEncoder
# except ModuleNotFoundError:
#     SentenceEncoder = NotInstalledError("SentenceEncoder", "sentence-tfm")

__all__ = [
    "SentenceEncoder"
]