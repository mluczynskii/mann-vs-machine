import csv 
import re
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

PATH="dataset.csv"
WINDOW_SIZE=2

def tokenize(text):
  regex = re.compile('[^a-zA-Z#@ ]')
  return regex.sub('', text).lower().split()

class W2VDataset(Dataset):
  def __init__(self):
    self.data = self.__process_file(PATH, WINDOW_SIZE)

  def __fetch_lines(self, filename):
    with open(filename, "r") as input:
      reader = csv.DictReader(input, fieldnames=["text", "label"])
      for row in reader:
        yield row["text"]

  def __process_file(self, filename, context_window):
    data, vocab = [], set()
    for tweet in self.__fetch_lines(filename):
      tokens = tokenize(tweet)
      n_tokens = len(tokens)
      for i, word in enumerate(tokens):
        vocab.add(word)
        left, right = max(0, i - context_window), min(n_tokens, i + context_window + 1)
        for j in range(left, right):
          if i != j:
            data.append((word, tokens[j]))
    self.vocab = {word: idx for idx, word in enumerate(vocab)}
    def to_idx(p):
      word, context = p 
      return self.vocab[word], self.vocab[context]
    return [to_idx(p) for p in data]
  
  def get_vocab(self):
    return self.vocab

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]

class SkipGram(nn.Module):
  def __init__(self, vocab_size, embed_size, negative_samples=10):
    super(SkipGram, self).__init__()
    self.vocab_size = vocab_size
    self.word_embed = nn.Embedding(vocab_size, embed_size)
    self.ctx_embed = nn.Embedding(vocab_size, embed_size)
    self.log_sigmoid = nn.LogSigmoid()
    self.negative_samples = negative_samples

  def forward(self, word, context, negative_samples):
    x = self.word_embed(word)
    y, neg = self.ctx_embed(context), self.ctx_embed(negative_samples)
    positive = self.log_sigmoid(torch.sum(x * y, dim=1))
    negative = self.log_sigmoid(-torch.bmm(neg, x.unsqueeze(2)).squeeze(2))
    return -(positive + negative.sum(1)).mean()
  
  def __get_negative_samples(self, words):
    result = np.empty(shape=(len(words), self.negative_samples))
    for i, exclude_idx in enumerate(words):
      iter = [idx for idx in range(0, self.vocab_size) if idx != exclude_idx.item()]
      result[i] = np.random.choice(iter, self.negative_samples, replace=False)
    return result
  
  def run_training(self, dataloader, num_epochs=10, learning_rate=0.01):
    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
      total_loss = 0
      for word_batch, context_batch in tqdm(dataloader, desc=f"Training the model, epoch={epoch+1}/{num_epochs}", leave=False):
        words = word_batch.long()
        contexts = context_batch.long()
        negative_samples = torch.LongTensor(self.__get_negative_samples(words))
        optimizer.zero_grad()
        loss = self(words, contexts, negative_samples)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      print(f"Epoch {epoch+1}: loss={total_loss/len(dataloader)}")

  def save_model(self, filename):
    torch.save(self, filename)

dataset = W2VDataset()
vocab = dataset.get_vocab()

def load_model(filename):
  model = torch.load(filename, weights_only=False)
  for param in model.parameters():
    param.requires_grad = False 
  model.eval()
  return model

def embed_sentence(sentence, model):
  tokens = torch.tensor([vocab[word] for word in tokenize(sentence) if word in vocab.keys()])
  embed = model.word_embed(tokens).mean(dim=0)
  print(f"Sentence embedding={embed}, shape={embed.shape}")
  return embed

  


