# E5 Embedder

Provides the E5 model

**Inputs**

- none

**Outputs**

- Embedder: an embedding model for Text Embedder or Reference Library

Provides the English `intfloat/e5-small-v2` model through Hugging Face Transformers. Vectors are
L2-normalized. The model is downloaded on first use and runs on the GPU when CUDA is available. The widget has no
settings.
