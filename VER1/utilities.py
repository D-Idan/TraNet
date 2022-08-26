import torch
from torch import nn, Tensor
import math

from params import max_mem_len

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    dev = torch.device("cpu")

# PositionalEncoding module injects some information about
# the relative or absolute position of the tokens in the sequence.
# The positional encodings have the same dimension as the embeddings so that the two can be summed.
# Here, we use sine and cosine functions of different frequencies.

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(d_model).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, max_mem_len, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(d_model, 1, max_mem_len)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.squeeze(pe)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        #x = x + self.pe[:x.size(0)]
        # pos_emb = self.pe[:x.size(0),:, :x.size(1)]
        pos_emb = self.pe[-2:, :x.size(1)]
        # pos_emb = torch.squeeze(pos_emb.to(dev))
        # pos_emb = pos_emb.unsqueeze(0).T
        x = torch.cat([pos_emb.to(dev), x])
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def calc_trace(A):
    l = A.shape[0]
    out = torch.empty(l)
    for k in range(l):
        out[k] = torch.trace(A[k,])
    return out

if __name__ == "__main__":
    r = PositionalEncoding(8)
    r.forward(torch.rand((10,1,8)))

    r = generate_square_subsequent_mask(8)

    print("FINISH!!")