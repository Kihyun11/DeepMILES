# model_cnn_only.py
import torch
import torch.nn as nn


class CNNOnly(nn.Module):
    def __init__(self, input_channels=6, num_classes=6, conv_kernels=64, dropout=0.5):
        super(CNNOnly, self).__init__()

        # Input: (batch, 1, seq_len, input_channels)
        self.features = nn.Sequential(
            nn.Conv2d(1, conv_kernels, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(conv_kernels, conv_kernels, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(conv_kernels, conv_kernels, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
            nn.Conv2d(conv_kernels, conv_kernels, kernel_size=(5, 1), padding=(2, 0)),
            nn.ReLU(),
        )

        # Global average pooling over (seq_len, sensor_dim)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(conv_kernels, num_classes)

    def forward(self, x, return_embeddings=False):
        """
        x: (batch, 1, seq_len, 6)
        """
        x = self.features(x)                  # (B, K, T, C)
        x = self.global_pool(x)               # (B, K, 1, 1)
        x = x.view(x.size(0), -1)             # (B, K)

        emb = self.dropout(x)

        if return_embeddings:
            return emb

        logits = self.classifier(emb)         # (B, num_classes)
        return logits