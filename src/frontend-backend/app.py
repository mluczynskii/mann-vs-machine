from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from logreg import LogisticRegressionModel
import torch
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = LogisticRegressionModel()

model_data = torch.load('logRegModel.pth')

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


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
