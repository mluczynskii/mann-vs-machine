from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import uvicorn
import os
from src.models.logreg import LogisticRegressionModel

app = FastAPI()
templates = Jinja2Templates(directory="src/frontend_backend/templates")

model = LogisticRegressionModel()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logRegModel.pth')
model_data = torch.load(model_path)

model.beta = model_data['beta'].numpy()
model.vocabulary = model_data['vocabulary']
model.dictionary = model_data['dictionary']

class TweetRequest(BaseModel):
    tweet: str

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify/")
def classify_tweet(request: TweetRequest):
    processed_tweet = model.preprocess_test_set([request.tweet], custom=True)
    classification = model.classify(processed_tweet)
    is_ai_generated = bool(classification[0])
    print(classification)
    return {"tweet": request.tweet, "is_ai_generated": is_ai_generated}
