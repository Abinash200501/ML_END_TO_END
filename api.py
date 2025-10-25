from fastapi import FastAPI
import uvicorn
import pickle
from fastapi.staticfiles import StaticFiles
from message import Message
import torch
from pathlib import Path
import subprocess
from fastapi.responses import FileResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

model_path = Path("saved_model") / "model.pkl"

if not model_path.exists():
    print("Model not found locally...pulling it from DVC")
    subprocess.run(["dvc","pull",str(model_path)], check=True)


model_data = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)


model = model_data['model']
tokenizer = model_data['tokenizer']

@app.get("/")
def home():
    return FileResponse("static/index.html")

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

