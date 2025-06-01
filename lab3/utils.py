from vocab import Vocab
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Embedding


def get_embedding_matrix(vocab: Vocab, dim: int = 300, freeze: bool = True, path= None):
    matrix = torch.normal(mean=0, std=1, size=(len(vocab.itos), dim))
    matrix[0] = torch.zeros(dim)
    
    if path is not None:
        with open(path, encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                token, values = parts[0], parts[1:]
                if token in vocab.stoi and len(values) == dim:
                    matrix[vocab.stoi[token]] = torch.tensor([float(x) for x in values], dtype=torch.float32)

    return Embedding.from_pretrained(matrix, padding_idx=0, freeze=freeze)


def pad_collate_fn(batch, pad_index=0):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(text) for text in texts])
    
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=pad_index)
    labels = torch.stack(labels).squeeze(1)
    
    return texts_padded, labels, lengths


def build_frequency_dict(instances):
    word_freqs = {}
    label_freqs = {}

    for instance in instances:
        for word in instance.text:
            word_freqs[word] = word_freqs.get(word, 0) + 1
        label = instance.label
        label_freqs[label] = label_freqs.get(label, 0) + 1

    return word_freqs, label_freqs