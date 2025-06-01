from torch import tensor
import torch

class Vocab:
    def __init__(self, frequencies: dict, max_size: int = -1, min_freq: int = 0, is_label=False):     
        self.pad = "<PAD>"
        self.unk = "<UNK>"
        
        if is_label:
            self.stoi = {"positive": 0, "negative": 1}
            self.itos = {0: "positive", 1: "negative"}
        else:
            self.itos = {0: self.pad, 1: self.unk}
            self.stoi = {self.pad: 0, self.unk: 1}
            
            for word, freq in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
                if max_size == -1 or max_size > len(self.itos):
                    if min_freq <= freq:
                        index = len(self.itos)
                        self.stoi[word] = index
                        self.itos[index] = word
                    else:
                        continue
                else:
                    break
            
    def encode(self, tokens):
        if isinstance(tokens, str):
            return torch.tensor(self.stoi.get(tokens, 1), dtype=torch.int64)
        else:
            indexes = []
            for token in tokens:
                if token in self.stoi:
                    indexes.append(self.stoi[token])
                else:
                    indexes.append(1)
            return tensor(indexes, dtype=torch.int64)