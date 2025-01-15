#!/usr/bin/env python3
import os 
import csv

AI_FILE = "ai-tweets.csv"
HUMAN_FOLDER = "data"

dataset = set()

for filename in os.listdir("data/"):
  with open(f"{HUMAN_FOLDER}/{filename}", "r") as input_human:
    for tweet in input_human:
      dataset.add((tweet, 0))

with open(AI_FILE, "r") as input_ai:
  for tweet in input_ai:
    dataset.add((tweet, 1))

with open("dataset.csv", "w", newline='') as output:
  writer = csv.DictWriter(output, fieldnames=['txt', 'lbl'], quotechar="'")
  for (tweet, label) in dataset:
    writer.writerow({"txt": ' '.join(tweet.split()), "lbl": label})