#!/usr/bin/env python3
import os 

dataset = set()

for filename in os.listdir("data/"):
  with open(f'data/{filename}', "r") as input:
    for tweet in input:
      dataset.add(tweet)

with open("tweets.csv", "w") as output:
  for tweet in dataset:
    output.write(tweet)