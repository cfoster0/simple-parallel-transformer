# simple-parallel-transformer
As it says on the tin, this repo has a simple implementation of a Transformer model, with some additional improvements. The purpose is mainly pedagogical.

## Design
More specifically, its block has the following modifications:
* Cogview's Sandwich Layer Norm, which puts a LayerNorm at the start and end of the block.
* Single-block design from Gated Attention Unit, by way of Mamba. Instead of a separate attention and feedforward layer, combines them in parallel as a gated attention layer, with the gating being passed into a SiLu/SwiGLU activation function. Also expands the internal dimension to be larger than the residual dimension.
* Smeared keys on each head, to facilitate learning of previous-token heads, induction heads, and n-grams.
* Per-head, data-dependent attention biasing. This is done by a projection -> sigmoid, cumulative sum to produce "absolute positions", and then a subtraction to get "relative positions" biases for each query-key pair, similar to ALiBi.

## References
* Sandwich Layer Norm - https://arxiv.org/abs/2105.13290
* GAU - https://arxiv.org/abs/2202.10447
* Mamba - https://arxiv.org/abs/2312.00752
* Smeared Key - https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
* ALiBi - https://arxiv.org/abs/2108.12409
