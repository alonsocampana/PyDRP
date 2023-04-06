from torch import nn

class ProteinConvPooling(nn.Module):
    def __init__(self,
                 init_dim = 1024,
                 hidden_dim = 512,
                 output_dim = 256,
                 n_groups = 8,
                 p_dropout_1 = 0.4,
                 p_dropout_2 = 0.4):
        super().__init__()
        self.conv_attn = nn.Sequential(nn.Conv1d(in_channels = init_dim,
                                                 out_channels=hidden_dim,
                                                 kernel_size = 6,
                                                 padding="same",
                                                 groups = n_groups),
              nn.ReLU(),
              nn.Dropout(p_dropout_1),
              nn.Conv1d(in_channels = hidden_dim,
                        out_channels=1,
                        kernel_size = 1,
                        stride = 1),)
        self.conv_seq = nn.Sequential(nn.Conv1d(in_channels = init_dim,
                                                out_channels=hidden_dim,
                                                kernel_size = 6,
                                                padding="same",
                                                groups = n_groups),
              nn.ReLU(),
              nn.Dropout(p_dropout_2),
              nn.Conv1d(in_channels = hidden_dim, out_channels=output_dim, kernel_size = 1, stride = 1),)
    def forward(self, x):
        a =  (self.conv_attn(x.transpose(1, 2)).squeeze()).softmax(-1)
        v = self.conv_seq(x.transpose(1, 2)).squeeze().transpose(1, 2)
        return a.unsqueeze(-1).mul(v).sum(axis=1)