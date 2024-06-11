# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/embedding.py
import math
import mlx.core as mx
import mlx.nn as nn
class TokenEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(p=dropout)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

    @property
    def weight(self) -> mx.array:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> mx.array:
        return self.word_embeddings.weight[index : index + 1]

    def __call__(self, x: mx.array):
        x = self.word_embeddings(x)
        x = self.dropout(x)
        return x


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = mx.array(1.0) 
        if not alpha: self.freeze(keys=["alpha"])
        self.dropout = nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(mx.zeros([1,4000]))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.shape[1] >= x.shape[1]:
                if self.pe.dtype != x.dtype:
                    self.pe = self.pe.astype(dtype=x.dtype)
                return
        pe = mx.zeros(x.shape[1], self.embedding_dim)
        if self.reverse:
            position = mx.expand_dims(mx.arange(
                x.shape[1] - 1, -1, -1.0, dtype=mx.Dtype.float32
            ),1)
        else:
            position = mx.expand_dims(mx.arange(0, x.shape[1], dtype=mx.Dtype.float32))
        div_term = mx.exp(
            mx.arange(0, self.embedding_dim, 2, dtype=mx.Dtype.float32)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        pe = mx.expand_dim(pe,0)
        self.pe = pe

    def __call__(self, x: mx.array) -> mx.array:
        self.extend_pe(x)
        output = mx.expand_dims(x,-1) if x.ndim == 2 else x
        output = output * self.x_scale + self.alpha * self.pe[:, : x.shape[1]]
        return self.dropout(output)
