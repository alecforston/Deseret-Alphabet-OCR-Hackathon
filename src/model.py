import torch.nn as nn

class CRNN(nn.Module):
   
    def __init__(self, num_classes, cnn_channels=[64, 128, 256, 512, 512],
                 hidden_size=256, num_layers=2, dropout=0.1):
        super(CRNN, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # CNN backbone
        self.cnn = self._make_cnn(cnn_channels)
        
        # RNN input size is the number of features from CNN
        rnn_input_size = cnn_channels[-1]
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            rnn_input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
        # Adaptive pooling to ensure height becomes 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
    
    def _make_cnn(self, channels):
        """Create CNN backbone"""
        layers = []
        in_channels = 1
        
        # First conv block
        layers.append(nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Subsequent conv blocks
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU(inplace=True))
            
            if i < 2:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # Rectangular pooling to preserve width
                layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: Input images (batch, 1, H, W)
        
        Returns:
            Output logits (seq_len, batch, num_classes)
        """
        # CNN feature extraction
        conv = self.cnn(x)
        
        # Apply adaptive pooling to ensure height is 1
        conv = self.adaptive_pool(conv)
        
        # Prepare for RNN
        b, c, h, w = conv.size()
        
        # Collapse height dimension
        conv = conv.squeeze(2)
        
        # Permute to (W', batch, channels) for RNN
        conv = conv.permute(2, 0, 1)
        
        # RNN
        rnn_out, _ = self.rnn(conv)
        
        # Fully connected
        seq_len, batch, hidden = rnn_out.size()
        rnn_out = rnn_out.view(seq_len * batch, hidden)
        output = self.fc(rnn_out)
        output = output.view(seq_len, batch, -1)
        
        return output

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)