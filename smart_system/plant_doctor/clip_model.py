import torch
from PIL import Image

try:
    import clip
except ImportError:
    clip = None

class CLIPModel:
    def __init__(self):
        if clip is None:
            raise ImportError("OpenAI CLIP is not installed")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self.text_tokens = None

        # Knowledge base
        self.labels = [
            "tomato leaf with early blight disease",
            "tomato leaf with septoria leaf spot",
            "potato leaf with late blight",
            "healthy green plant leaf",
            "plant leaf with fungal infection",
            "plant leaf with bacterial disease",
            "plant leaf with virus infection"
        ]

    def _ensure_loaded(self):
        if self.model is None:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.text_tokens = clip.tokenize(self.labels).to(self.device)

    def predict(self, image_path):
        self._ensure_loaded()
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.text_tokens)

            similarity = (image_features @ text_features.T).softmax(dim=-1)

        values, indices = similarity[0].topk(3)

        results = []
        for i, idx in enumerate(indices):
            results.append({
                "label": self.labels[idx],
                "confidence": float(values[i].item())
            })

        return results