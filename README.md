# llm-interpretability

A from-scratch mechanistic interpretability experiment — training a Sparse Autoencoder (SAE) on the internal activations of Qwen2-0.5B, running entirely on a laptop CPU.

-----

## What This Is

This project hooks into the MLP outputs of layers 9–14 of Qwen2-0.5B, collects token-level activations across a text corpus, and trains a Sparse Autoencoder to decompose those activations into interpretable sparse features. The goal is to find human-readable concepts encoded inside the model’s internals — without using any pre-built interpretability library.

-----

## Results

|Metric             |Value               |
|-------------------|--------------------|
|Model              |Qwen2-0.5B          |
|Layers hooked      |9–14 (40–58% depth) |
|Tokens collected   |18,639              |
|SAE hidden dim     |3,584 (4× expansion)|
|Sparsity (λ=0.5)   |**86.6%**           |
|Reconstruction loss|**0.0058**          |
|Training hardware  |CPU only            |

### Sample features found

|Neuron|Firing token|Label                                                     |
|------|------------|----------------------------------------------------------|
|0     |trade       |NHL trade and contract language                           |
|3459  |charge      |Formal legal authority language with subordinate clauses  |
|2659  |1           |Capitalized poetic title words in Japanese media          |
|1289  |mission     |Military mission terminology in game design context       |
|3438  |citizens    |Collective civic nouns in 19th century political discourse|
|3551  |the         |Definite article in formal governmental possession claims |

-----

## How It Works

### 1. Activation Collection

Register forward hooks on MLP outputs at layers 9–14. Run 338 sentences from wikitext-103 through Qwen2-0.5B and collect one 896-dim vector per token per layer. Flatten into a `(18639, 896)` activation matrix.

```python
captured = {}

def get_hook(layer_id):
    def hook(module, input, output):
        captured[layer_id] = output.cpu().detach()
    return hook

for n in [9, 10, 11, 12, 13, 14]:
    model.model.layers[n].mlp.register_forward_hook(get_hook(n))
```

### 2. SAE Architecture

```
Encoder: Linear(896 → 3584) + ReLU
Decoder: Linear(3584 → 896)
Loss:    MSE(reconstruction, input) + λ * hidden.abs().mean()
```

```python
class SAE(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.encoder = nn.Linear(in_dim, out_dim)
        self.decoder = nn.Linear(out_dim, in_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.relu(self.encoder(x))
        reconstruction = self.decoder(hidden)
        return reconstruction, hidden
```

### 3. Feature Interpretation

For each neuron, find the top-k token positions where it activated most strongly. Map token positions back to source sentences using cumulative sequence length boundaries. Read the sentences to identify the concept.

```python
# build boundary map
lengths = torch.tensor([t.shape[1] for t in store[9]])
boundaries = torch.cumsum(lengths, dim=0)

# find top activating tokens per neuron
for neuron_idx in range(hidden.shape[1]):
    topk = torch.topk(hidden[:, neuron_idx], 3)
    sentences = [(boundaries > i).nonzero()[0].item() for i in topk.indices]
```

-----

## Key Finding — Corpus Bias

The top-10 most frequently appearing sentences across all 3,584 neurons were sentences 1–41, almost entirely from two Wikipedia articles — Valkyria Chronicles III and the Little Rock Arsenal.

With only 338 sentences, two dominant topics skew the entire feature atlas. The SAE learns domain-specific features rather than general linguistic ones.

**Features are only as general as the data they’re trained on.**

-----

## What I Learned

- Hook functions must not return anything — returning the output accidentally replaces the module’s forward output
- `captured` dict only holds the last sentence’s activations — a separate `store` dict is needed to accumulate across sentences
- Registering hooks multiple times across kernel restarts causes KeyErrors — always restart and re-register fresh


-----

## Setup

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
```

-----

## Reference

- [Towards Monosemanticity — Anthropic](https://transformer-circuits.pub/2023/monosemantic-features)
- [Qwen-Scope — Alibaba](https://qwen.ai/blog?id=qwen-scope) — released the same week this was built