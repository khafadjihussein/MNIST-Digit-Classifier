from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from model import TinyCNN

class Predictor:
    """
    Loads a trained TinyCNN and predicts digit from PIL image.
    """
    def __init__(self, weights: str = "mnist_cnn.pt", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyCNN().to(self.device)
        ckpt_path = Path(weights)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model weights not found at {weights}. Train the model first.")
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.tfm = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> Tuple[int, float, List[float]]:
        """
        Args:
            pil_image: PIL Image (any mode/size)
        Returns:
            digit: int [0-9]
            confidence: float [0,1]
            probs: list of float (len=10)
        """
        img = pil_image.convert("L")
        img = ImageOps.invert(img)
        img = self.tfm(img)
        img = img.unsqueeze(0).to(self.device)
        logits = self.model(img)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        digit = int(probs.argmax())
        confidence = float(probs[digit])
        return digit, confidence, probs.tolist()
