from vocab import Vocab
from dataset import NLPDataset
from utils import get_embedding_matrix, pad_collate_fn, build_frequency_dict
from torch.utils.data import DataLoader

temp_dataset = NLPDataset(text_vocab=None, label_vocab=None, path="data/train.csv")

word_freqs, label_freqs = build_frequency_dict(temp_dataset.instances)

text_vocab = Vocab(word_freqs, max_size=-1, min_freq=0)
label_vocab = Vocab(label_freqs, is_label=True)

embedding = get_embedding_matrix(vocab=text_vocab, dim=300, freeze=True, path="data/glove.txt",)

train_dataset = NLPDataset(text_vocab=text_vocab, label_vocab=label_vocab, path="data/train.csv")

instance = train_dataset.instances[3]
instance_text = instance.text
instance_label = instance.label
print(f"Text: {instance_text}")
print(f"Label: {instance_label}")

numericalized_text = text_vocab.encode(instance_text)
numericalized_label = label_vocab.encode(instance_label)

print(f"Numericalized text: {numericalized_text}")
print("Numericalized label:", repr(numericalized_label))

print(len(text_vocab.itos))

batch_size = 2
shuffle = False
train_dataset = NLPDataset(text_vocab=text_vocab, label_vocab=label_vocab, path="data/train.csv")
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
                    collate_fn=lambda batch: pad_collate_fn(batch, pad_index=text_vocab.stoi["<PAD>"]))
 
texts, labels, lengths = next(iter(train_dataloader))
print(f"Texts: {texts}")
print(f"Labels: {labels}")
print(f"Lengths: {lengths}")