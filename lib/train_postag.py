import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from conllu import parse_incr

PAD_ID = 0

n_epochs = 10
lr = 0.01
batch_size = 32


class RNN(nn.Module):
    def __init__(
            self,
            input_size,
            embedding_size,
            hidden_size,
            output_size,
            p=0.2
    ):
        super(RNN, self).__init__(padding_idx=PAD_ID)

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding_layer = nn.Embedding(
            input_size, embedding_size, padding_idx=PAD_ID
        )
        self.gru = nn.GRU(
            embedding_size, hidden_size, batch_first=True, bias=False
        )
        self.dropout = nn.Dropout(p=p)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        e = self.embedding_layer(x)
        h = self.gru(e)[1].squeeze(dim=0)

        return self.linear(self.dropout(h))


def fit(model, data):

    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True
    )

    for epoch in n_epochs:
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            x, y_true = batch

            y_pred = model(x)
            loss = criterion(y_true, y_pred)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")


def perf(model, data):

    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        dataset=data, batch_size=batch_size, shuffle=True
    )

    for batch in train_loader:
        x, y_true = batch
        y_pred = model(x)

        loss = criterion(y_true, y_pred)

        print(f"Loss: {loss.item():.4f}")


def pad_tensor(X, max_len):
    res = torch.full((len(X), max_len), 0)
    for (i, row) in enumerate(X):
        x_len = min(max_len, len(X[i]))
        res[i, :x_len] = torch.LongTensor(X[i][:x_len])
    return res


def read_corpus(
        filename, wordvocab, tagvocab, train_mode=True, batch_mode=True
):

    if train_mode:
        wordvocab = collections.defaultdict(lambda: len(wordvocab))
        wordvocab["<PAD>"]
        wordvocab["<UNK>"]  # Create special token IDs
        tagvocab = collections.defaultdict(lambda: len(tagvocab))
    words, tags = [], []
    with open(filename, 'r', encoding="utf-8") as corpus:
        for line in corpus:
            fields = line.strip().split()
            tags.append(tagvocab[fields[0]])
            if train_mode:
                words.append([wordvocab[w] for w in fields[1:]])
            else:
                words.append(
                    [wordvocab.get(w, wordvocab["<UNK>"]) for w in fields[1:]]
                )
    if batch_mode:
        dataset = TensorDataset(
            pad_tensor(words, 40), torch.LongTensor(tags)
        )
        return DataLoader(dataset, batch_size=32, shuffle=train_mode), wordvocab, tagvocab
    else:
        return words, tags, wordvocab, tagvocab


if __name__ == "__main__":
    for sent in parse_incr(open("../sequoia/sequoia-ud.parseme.frsemcor.simple.small", encoding="UTF-8")):
        print([(tok["form"], tok["upos"]) for tok in sent])
        # print(" ".join(tok["upos"] for tok in sent))
        assert False
