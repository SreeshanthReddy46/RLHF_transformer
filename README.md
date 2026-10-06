# Mini RLHF Transformer Demo

This project is a small, self-contained demonstration of how a language model can be optimized with a reward signal. It uses PyTorch to:

1. Create a tiny causal Transformer language model.
2. Generate tokens one at a time from a prompt.
3. Score generated tokens with a hand-written reward function.
4. Update the policy model using a policy-gradient loss.
5. Print generations before and after training.

This is an educational RLHF-style example, not a complete production RLHF system. It does not use a dataset, tokenizer, pretrained model, human preference labels, or a separately trained reward model.

## Requirements

- Python 3.9 or newer
- PyTorch

Install PyTorch if it is not already installed:

```bash
pip install torch
```

## Run the program

From this folder, run:

```bash
python main.py
```

On Windows, the command can also be written with the full Python path:

```powershell
C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe main.py
```

The program prints three sections:

- Initial generations before policy training
- Training progress at selected epochs
- Final generations after policy training

## Vocabulary

The model uses integer token IDs defined in `VOCAB`:

| ID | Token |
|---:|---|
| 0 | `<PAD>` |
| 1 | `<START>` |
| 2 | `<good` |
| 3 | `bad` |
| 4 | `neutral` |
| 5 | `code` |
| 6 | `bug` |

The initial prompt is:

```python
prompt = torch.tensor([[1]])
```

ID `1` is the `<START>` token. The model generates four additional tokens for each sequence.

## Model architecture

### `CausalSelfAttention`

This class implements masked self-attention:

1. The input is projected into queries, keys, and values with `qkv_proj`.
2. Each tensor is split into eight attention heads.
3. Attention scores are calculated with scaled dot-product attention.
4. A lower-triangular mask prevents a token from looking at future tokens.
5. The heads are joined and projected back to the model dimension.

The input shape is:

```text
(batch_size, sequence_length, embedding_dimension)
```

With the default settings, the embedding dimension is `64` and there are `8` heads, so each head has dimension `8`.

### `TransformerBlock`

Each block contains:

- Layer normalization
- Causal self-attention
- A residual connection
- A feed-forward MLP
- A second residual connection

The `return x` at the end of `forward()` is required. Without it, the next Transformer block receives `None`, producing the PyTorch error:

```text
TypeError: layer_norm(): argument 'input' must be Tensor, not NoneType
```

### `MiniTransformerLM`

The language model combines:

- Token embeddings
- Positional embeddings
- Two Transformer blocks
- Final layer normalization
- A linear language-model head

The output is a tensor of logits with shape:

```text
(batch_size, sequence_length, vocabulary_size)
```

Each position contains one score for every possible token.

## Text generation

`generate_sequence()` performs autoregressive generation:

1. Run the model on the tokens generated so far.
2. Select the logits for the final position.
3. Convert logits into probabilities with softmax.
4. Sample the next token with `Categorical`.
5. Append that token to the sequence.
6. Repeat four times.

The function returns:

- `curr_tokens`: the prompt plus generated tokens
- `sampled_actions`: only the newly generated token IDs
- `log_probs`: log-probabilities of the sampled actions

Sampling is random, but `torch.manual_seed(42)` makes the initial random state reproducible.

## Reward function

`reward_function()` assigns a score independently for every generated token:

| Token | Reward |
|---|---:|
| `<good` | `+1.5` |
| `code` | `+1.0` |
| `bad` | `-1.5` |
| `bug` | `-1.0` |
| `<PAD>`, `<START>`, `neutral` | `0.0` |

Rewards are added together. For example:

```text
<good code bug neutral
```

receives:

```text
1.5 + 1.0 - 1.0 + 0.0 = 1.5
```

## Policy and reference models

Two identical models are created:

- `policy_model`: the model being trained
- `ref_model`: a frozen copy used as the reference policy

The optimizer updates only `policy_model`. The reference model is placed in evaluation mode and its parameters have gradients disabled.

## Training objective

For each epoch, the program:

1. Generates a sequence with the current policy.
2. Calculates the raw reward.
3. Runs both the policy and reference models on the generated sequence.
4. Calculates a KL-divergence-style penalty.
5. Combines reward and penalty into `net_reward`.
6. Uses the policy-gradient loss:

```python
loss = -action_log_probs.sum() * net_reward
```

7. Backpropagates and updates the policy model with AdamW.

The loop runs for 100 epochs with a learning rate of `1e-3`. Progress is printed at epochs `1`, `20`, `40`, `60`, `80`, and `100`.

## Important implementation note

The current code defines `p_ref` from `gen_policy_logits`:

```python
p_ref = F.softmax(gen_policy_logits, dim=-1)
```

Therefore, `p_policy` and `p_ref` are identical and the KL value is effectively zero. This README documents the code as it currently exists; changing that line to use `gen_ref_logits` would activate the intended reference-model penalty and would be a separate code change.

## Main constants to experiment with

These values can be changed in `main.py`:

```python
d_model=64       # embedding size
n_heads=8        # attention heads
n_layers=2       # Transformer blocks
max_len=32       # maximum supported sequence length
gen_len=4        # generated tokens per sequence
lr=1e-3          # optimizer learning rate
beta_kl=0.15     # KL penalty coefficient
```

Keep `d_model` divisible by `n_heads`, because each attention head uses:

```text
head_dim = d_model / n_heads
```

## Limitations

- The vocabulary is extremely small and manually defined.
- The model starts with random weights.
- The reward function is a simple Python function, not a learned reward model.
- Only one prompt is used.
- The batch size is one.
- There is no checkpoint saving or loading.
- Generated output is token IDs mapped directly to toy vocabulary strings.
- This example demonstrates the mechanics of policy optimization, not robust language-model alignment.
