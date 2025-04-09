import torch
import torch.nn as nn

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    """
    Compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len=512, device=None):
        """
        Constructor of sinusoid encoding class.

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # Same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        :param x: Input tensor of shape [batch_size, seq_len, d_model]
        :return: Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.size()
        return x + self.encoding[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, seq_len, max_len=512, device=None):
        super(Transformer, self).__init__()

        # Initialize parameters
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Linear layers for input and output embedding
        self.input_fc = nn.Linear(input_size, d_model)
        self.output_fc = nn.Linear(input_size, d_model)

        # Positional Encoding
        self.pos_emb = PositionalEncoding(d_model=d_model, max_len=max_len, device=self.device)

        # Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
            device=self.device
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)

        # Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
            device=self.device
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=5)

        # Final fully connected layers
        self.fc1 = nn.Linear(seq_len * d_model, d_model)
        self.fc2 = nn.Linear(d_model, output_size)

    def forward(self, x):
        # # Extract target sequence from input
        # y = x[:, -self.output_size:, :]

        # Input embedding
        x = self.input_fc(x)  # [batch_size, seq_len, d_model]
        x = self.pos_emb(x)  # Add positional encoding
        x = self.encoder(x)  # Pass through encoder

        # # Flatten and pass through final FC layers
        # x = x.flatten(start_dim=1)  # [batch_size, seq_len * d_model]
        # x = self.fc1(x)  # [batch_size, d_model]
        # out = self.fc2(x)  # [batch_size, output_size]

        return x

