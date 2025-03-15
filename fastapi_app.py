from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
from io import BytesIO

app = FastAPI()

model_path= "/content/drive/MyDrive/laptop_price_model.pkl"
model= joblib.load(model_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
  content = await file.read()
  df = pd.read_csv(BytesIO(content))
  predictions = model.predict(df)
  return {"prediction": predictions.tolist()}