# MNIST Digit Classifier

A minimal, production-style MNIST digit classifier using PyTorch and FastAPI.

- **Stack:** Python 3.11, PyTorch, torchvision, FastAPI, Pydantic v2, Uvicorn, Pillow, numpy
- **Features:** Deterministic training, best checkpoint saving, REST API for predictions, type hints, clean structure.

## Setup & Usage

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (downloads MNIST, saves mnist_cnn.pt)
python train.py

# 4. Serve the API (open http://127.0.0.1:8000/docs)
uvicorn api:app --reload
```

## Example: Predict via API

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@tests/data/digit.png"
```

- Response: `{"digit": 3, "confidence": 0.99}`

## Image Format

- Input: 28×28 PNG, white digit on black background (like MNIST).
- The API **automatically inverts colors** if you upload a black digit on white.

## Testing

```powershell
pytest -q
```

## Acceptance Checklist

- [x] `python train.py` reaches ≥97% val accuracy in ~5 epochs.
- [x] `uvicorn api:app --reload` serves `/docs` and `/healthz`.
- [x] Posting a 28×28 PNG to `/predict` returns digit & confidence.
- [x] `pytest -q` passes all tests.

## Optional Extensions

- Dockerfile for containerization
- ONNX export for model portability
- React UI for drawing digits

---

**Note:** For best results, use 28×28 grayscale PNGs. The API will invert colors as needed.

---

## Frontend & Docker

### Local Development

**Backend**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
python train.py
uvicorn api:app --reload
```

**Frontend**
```powershell
cd ui
copy .env.example .env
npm install
npm run dev
# open http://127.0.0.1:5173
```

# RUN SERVER (Backend + Frontend)

.\.venv\Scripts\Activate
uvicorn api:app --reload

cd ui
npm run dev

# Open in browser:
# http://127.0.0.1:5173        (Frontend UI)
# http://127.0.0.1:8000/docs   (API Docs)
# http://127.0.0.1:8000/healthz (Health Check)

### Docker Compose

```powershell
docker compose up --build
# frontend: http://127.0.0.1:8080
# backend:  http://127.0.0.1:8000/docs
```

**Note:** The API inverts colors to match MNIST (white digit on black).

#### Troubleshooting

- If you get CORS errors, check that the frontend is using the correct API URL and that the backend allows the origin.
- Large canvas images may take longer to upload; the API works best with 28×28 or small PNGs.

---
