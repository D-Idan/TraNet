
emsize = 200  # embedding dimension (d_model)
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
dim_feedforward_dec = 64 # dimension of the feedforward network model in nn.TransformerDecoder
nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder and for Decoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability

max_mem_len = 12 # 12mem * 3state = 36
n = m = 3