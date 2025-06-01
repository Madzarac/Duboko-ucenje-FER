from vocab import Vocab
from dataset import NLPDataset
from utils import get_embedding_matrix, pad_collate_fn, build_frequency_dict
from torch.utils.data import DataLoader
import torch
import numpy as np
from baseline import BaselineModel
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for texts, labels, _ in data_loader:
        texts, labels = texts.to(device), labels.float().to(device)

        model.zero_grad()
        logits = model(texts)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)



def evaluate(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for texts, labels, _ in data_loader:
            texts, labels = texts.to(device), labels.float().to(device)

            logits = model(texts)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)

    return avg_loss, acc, f1, cm

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    temp_dataset = NLPDataset(text_vocab=None, label_vocab=None, path="data/train.csv")

    word_freqs, label_freqs = build_frequency_dict(temp_dataset.instances)

    text_vocab = Vocab(word_freqs, max_size=-1, min_freq=1)
    label_vocab = Vocab(label_freqs, is_label=True)

    train_dataset = NLPDataset(text_vocab, label_vocab, path="data/train.csv")
    valid_dataset = NLPDataset(text_vocab, label_vocab, path="data/valid.csv")
    test_dataset  = NLPDataset(text_vocab, label_vocab, path="data/test.csv")

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=lambda b: pad_collate_fn(b, text_vocab.stoi["<PAD>"]))
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
                              collate_fn=lambda b: pad_collate_fn(b, text_vocab.stoi["<PAD>"]))
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=lambda b: pad_collate_fn(b, text_vocab.stoi["<PAD>"]))

    embedding = get_embedding_matrix(text_vocab, dim=300, path="data/glove.txt", freeze=True)
    model = BaselineModel(embedding).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_cm = evaluate(model, valid_loader, criterion, device)

        print(f"\nEpoch {epoch}:")
        print(f" train loss = {train_loss:.4f}")
        print(f" valid loss = {val_loss:.4f}, accuracy = {val_acc*100:.2f}%, f1 = {val_f1:.4f}")
        print(f" confusion matrix:\n{val_cm}")

    test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, criterion, device)
    print("\nTest performance:")
    print(f" loss = {test_loss:.4f}, accuracy = {test_acc*100:.2f}%, f1 = {test_f1:.4f}")
    print(f" confusion matrix:\n{test_cm}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7052020)
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
