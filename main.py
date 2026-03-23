from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("sentiment_model.pkl")

@app.get("/")
def home():
    return {"message": "Sentiment API is running!"}

@app.get("/predict")
def predict(text: str):
    prediction = model.predict([text])[0]

    if prediction == 1:
        result = "Positive 😊"
    else:
        result = "Negative 😡"

    return {"prediction": result}