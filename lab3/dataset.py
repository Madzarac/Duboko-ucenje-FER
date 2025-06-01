from torch.utils.data import Dataset
import pandas as pd
from instance import Instance
from vocab import Vocab
from pathlib import Path

class NLPDataset(Dataset):
    def __init__(self, text_vocab: Vocab, label_vocab: Vocab, path: Path):
        self.vocab_text = text_vocab
        self.vocab_labels = label_vocab
        self.instances = []

        data = pd.read_csv(path, header=None)
        for text, label in zip(data[0], data[1]):
            self.instances.append(Instance(text.strip().split(), label.strip()))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i]
        text = instance.text
        label = [instance.label]
        return self.vocab_text.encode(text), self.vocab_labels.encode(label)
