# src/model.py
import torch
import torch.nn as nn

class BaseCallerNet(nn.Module):
    def __init__(self, use_meta=True, meta_dim=5):
        """
        use_meta: подавать ли мета-признаки (quality, ratio, match flag)
        meta_dim: размерность мета-вектора
        """
        super().__init__()
        self.use_meta = use_meta

        # CNN для обработки сигнала (8 каналов: 4 fwd + 4 rev)
        self.conv = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU()
        )

        # Transformer для учета контекста
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.pool = nn.AdaptiveAvgPool1d(1)  # -> (batch, 128, 1)

        # Если используются мета-признаки — дополнительный MLP
        if use_meta:
            self.meta_fc = nn.Sequential(
                nn.Linear(meta_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32)
            )
            combined_dim = 128 + 32
        else:
            combined_dim = 128

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # A, C, G, T
        )

    def forward(self, x, meta=None):
        # x: (batch, 8, 240)
        x = self.conv(x)          # (batch, 128, 240)
        x = x.transpose(1, 2)     # (batch, 240, 128)
        x = self.transformer(x)   # (batch, 240, 128)
        x = x.transpose(1, 2)     # (batch, 128, 240)
        x = self.pool(x).squeeze(-1)  # (batch, 128)

        if self.use_meta and meta is not None:
            m = self.meta_fc(meta)    # (batch, 32)
            x = torch.cat([x, m], dim=1)  # (batch, 128+32)

        return self.fc(x)  # (batch, 4)