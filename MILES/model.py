import torch
import torch.nn as nn

class DeepConvLSTM(nn.Module):
    def __init__(self, input_channels=6, num_classes=5, conv_kernels=64, lstm_units=128):
        super(DeepConvLSTM, self).__init__()
        
        # 4 Convolutional Layers to extract spatial-temporal features
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, conv_kernels, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.Conv2d(conv_kernels, conv_kernels, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.Conv2d(conv_kernels, conv_kernels, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.Conv2d(conv_kernels, conv_kernels, kernel_size=(5, 1)),
            nn.ReLU()
        )
        
        # LSTM Layers
        # Input to LSTM: (batch, seq_len, features)
        self.lstm = nn.LSTM(conv_kernels * input_channels, lstm_units, num_layers=2, batch_first=True)
        
        # Final Classifier
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(lstm_units, num_classes)

    # def forward(self, x):
    #     # x shape: (batch, 1, seq_len, 6) -> 6 sensors (acc_x,y,z, gyro_x,y,z)
    #     x = self.conv_block(x)
        
    #     # Reshape for LSTM: (batch, new_seq_len, conv_kernels * 6)
    #     batch, kernels, seq_len, sensors = x.size()
    #     x = x.permute(0, 2, 1, 3).contiguous()
    #     x = x.view(batch, seq_len, kernels * sensors)
        
    #     x, (h_n, c_n) = self.lstm(x)
        
    #     # We take the output of the last time step
    #     x = self.dropout(x[:, -1, :])
    #     return self.classifier(x)
    
    def forward(self, x, return_embeddings: bool = False):
        """
        x: (batch, 1, seq_len, 6)
        return_embeddings:
          - False: returns logits (batch, num_classes)
          - True:  returns embedding vector (batch, lstm_units)
        """
        x = self.conv_block(x)

        batch, kernels, seq_len, sensors = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, seq_len, kernels * sensors)

        x, _ = self.lstm(x)

        emb = x[:, -1, :]          # (batch, lstm_units)
        emb = self.dropout(emb)    # dropout disabled automatically in eval()

        if return_embeddings:
            return emb

        return self.classifier(emb)