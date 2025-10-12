from fastapi import FastAPI
import uvicorn
import pickle
from message import Message
import torch
from pathlib import Path

app = FastAPI()

model_path = Path("saved_model") / "model.pkl"

with open(model_path, "rb") as f:
    model_data = pickle.load(f) 
model = model_data['model']
tokenizer = model_data['tokenizer']

@app.get("/")
def home():
    return {"Greeting:":"Welcome to the Spam classification"}

@app.post("/predict")
def predict(message: Message):
    print("Predict endpoint was hit!")

    device = torch.device("cpu")

    model.to(device)

    tokenized_input = tokenizer(message.message, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**tokenized_input)  

    logits = output.logits
    probs = torch.sigmoid(logits)
    prediction = (probs >= 0.5).long().item()

    if prediction == 1:
        return {"prediction": "Spam message"}
    else:
        return {"prediction": "Valid message"}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

