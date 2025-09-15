from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from infer import Predictor
from PIL import Image, UnidentifiedImageError
import io

app = FastAPI(title="MNIST Digit Classifier API")

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
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="Invalid image file.")
    pred = get_predictor()
    digit, confidence, _ = pred.predict(img)
    return Prediction(digit=digit, confidence=confidence)

@app.get("/healthz")
def healthz():
    return {"ok": True}
