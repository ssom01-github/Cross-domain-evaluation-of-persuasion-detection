import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.W2 = nn.Linear(hidden_dim * 2 + embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        combined = torch.cat((outputs, embedded), dim=2)
        Y = self.W2(combined).permute(0, 2, 1)
        Y = F.max_pool1d(Y, Y.size(2)).squeeze(2)
        return self.fc(Y)
