from nemo.collections.nlp.models import TextEmbeddingModel
import torch

class NeMoEmbedder:
    def __init__(self, model_name="st-ncls/bert-base-uncased-stsb"):
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed_texts(self, texts):
        """
        texts: List[str]
        returns: List[List[float]] embedding vectors
        """
        with torch.no_grad():
            embeddings = self.model.encode(texts, device=self.device)
        return embeddings.cpu().numpy()
