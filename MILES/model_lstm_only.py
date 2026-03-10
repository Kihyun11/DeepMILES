# model_lstm_only.py
import torch
import torch.nn as nn


class LSTMOnly(nn.Module):
    def __init__(self, input_channels=6, num_classes=6, lstm_units=128, num_layers=2, dropout=0.5):
        super(LSTMOnly, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, num_classes)

    def forward(self, x, return_embeddings=False):
        """
        x: (batch, 1, seq_len, 6)
        """
        x = x.squeeze(1)                      # (B, T, 6)
        x, _ = self.lstm(x)                   # (B, T, H)

        emb = self.dropout(x[:, -1, :])       # last timestep embedding

        if return_embeddings:
            return emb

        logits = self.classifier(emb)         # (B, num_classes)
        return logits