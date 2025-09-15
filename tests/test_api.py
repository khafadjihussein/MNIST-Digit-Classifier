import pytest
from fastapi.testclient import TestClient
from api import app
from pathlib import Path
from PIL import Image, ImageDraw
import io

client = TestClient(app)

def make_test_digit():
    """Create a 28x28 PNG with a white '3' on black."""
    img = Image.new("L", (28, 28), 0)
    draw = ImageDraw.Draw(img)
    draw.text((7, 4), "3", fill=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

def test_predict_valid():
    img_bytes = make_test_digit()
    resp = client.post("/predict", files={"file": ("digit.png", img_bytes, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert "digit" in data and "confidence" in data
    assert isinstance(data["digit"], int)
    assert 0 <= data["digit"] <= 9
    assert 0.0 <= data["confidence"] <= 1.0
