from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from infer import Predictor
from PIL import Image, UnidentifiedImageError
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MNIST Digit Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prediction(BaseModel):
    digit: int
    confidence: float

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        try:
            predictor = Predictor(weights="mnist_cnn.pt")
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
    return predictor

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        img.save("debug_input.png")  # Save uploaded image for debugging
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    pred = get_predictor()
    digit, confidence, _ = pred.predict(img)
    return Prediction(digit=digit, confidence=confidence)

@app.get("/healthz")
def healthz():
    return {"ok": True}
