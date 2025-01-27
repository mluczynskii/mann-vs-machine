from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
import uvicorn
import os
from src.models.naive_bayes import NaiveBayesModel

app = FastAPI()
templates = Jinja2Templates(directory="src/frontend_backend/templates")

model = NaiveBayesModel()

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'naiveBayesModel.pth')
model_data = torch.load(model_path)

model.n = model_data['n']
model.dictionary = model_data['dictionary']
model.custom_trained = True

class TweetRequest(BaseModel):
    tweet: str

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify/")
def classify_tweet(request: TweetRequest):
    processed_tweet = model.preprocess_set([request.tweet])
    classification = model.classify(processed_tweet)
    is_ai_generated = bool(classification[0])
    print(classification)
    return {"tweet": request.tweet, "is_ai_generated": is_ai_generated}
