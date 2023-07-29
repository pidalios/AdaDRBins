import torch
import torch.nn as nn
from transformer.swin_transformer import StageModule
# from swin_transformer import StageModule

class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)

        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x

class miniSwinEncoder(nn.Module):
    def __init__(self, *, hidden_dim=96, layers, heads, channels=32, head_dim=32, window_size=7, downscaling_factors=(4, 2), relative_pos_embedding=True):
        super().__init__()
        self.name = 'miniSwinEncoder'
        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0], 
                                downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim, 
                                window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim*2, layers=layers[1], 
                                downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim, 
                                window_size=window_size, relative_pos_embedding=relative_pos_embedding)
    def forward(self, x):
        x = self.stage1(x)
        # x = self.stage2(x)
        return x

class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)
