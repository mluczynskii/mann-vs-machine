## Utilities for creating the dataset
1. `scrape.py` - Webscraper code used to gather real-life tweets. Make sure to create a `/data` folder to store results, and `.env` file with your Twitter/X credentials (`USERNAME` and `PASSWORD`).
2. `gpt_neo_tweets.ipynb` - Jupyter notebook used to generate tweets using GPT-Neo model. Best to run it inside a Google Colab environment
3. `combine.py` - Simple utility script that combines every webscraper run with the ai-generated tweets, and assigns labels inside a `.csv` file. Just make sure to set global constants corresponding to correct filenames.
4. `embedding.py` + `parameters.pth` - Implementation and pre-trained SkipGram model for sentence vectorization.  