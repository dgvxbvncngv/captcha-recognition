import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, input_channel=1, hidden_size=512, num_layers=4):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 2)),
        )
        self.feature_height = 2
        self.cnn_output_size = 512 * self.feature_height

        self.rnn = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)                # (batch, 512, 2, 10)
        batch, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch, w, c * h)    # (batch, 10, 1024)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
