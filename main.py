import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads


        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        casual_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(casual_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=8, n_layers=2, max_len=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

torch.manual_seed(42)

VOCAB = {0: "<PAD>", 1: "<START>", 2: "<good", 3: "bad", 4: "neutral", 5: "code", 6: "bug"}
VOCAB_SIZE = len(VOCAB)

policy_model = MiniTransformerLM(vocab_size=VOCAB_SIZE)
ref_model = MiniTransformerLM(vocab_size=VOCAB_SIZE)

ref_model.load_state_dict(policy_model.state_dict())
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-3)
beta_kl = 0.15

def reward_function(generated_tokens):
    """
    Reward function (Simulating a reward model):
    - Rewards sequences containing "good" (2) and "code" (5)
    - Penalizes sequences containing "bad" (3) and "bug" (6)
    """
    score = 0.0
    for tok in generated_tokens:
        if tok == 2: score += 1.5
        elif tok == 5: score += 1.0
        elif tok == 3: score -= 1.5
        elif tok == 6: score -= 1.0
    return score

def generate_sequence(model, prompt_tokens, gen_len=4):
    """Autoregressive sampling with log-probabilities tracked."""
    curr_tokens = prompt_tokens.clone()
    sampled_actions = []
    log_probs = []


    for _ in range(gen_len):
        logits = model(curr_tokens)
        next_token_logits = logits[:, -1, :]

        probs = F.softmax(next_token_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        sampled_actions.append(action)
        log_probs.append(dist.log_prob(action))

        curr_tokens = torch.cat([curr_tokens, action.unsqueeze(1)], dim=1)

    return curr_tokens, torch.stack(sampled_actions, dim=1), torch.stack(log_probs, dim=1)


prompt = torch.tensor([[1]])

print("--- Initial Generations Before RLHF ---")
policy_model.eval()
with torch.no_grad():
    for _ in range(3):
        full_seq, _, _ = generate_sequence(policy_model, prompt, gen_len=4)
        tokens = [VOCAB[t.item()] for t in full_seq[0]]
        print("Generated:", " ".join(tokens[1:]))

print("\n--- Training RLHF Policy ---")
policy_model.train()

for epoch in range(1 ,101):
    full_seq, actions, action_log_probs = generate_sequence(policy_model, prompt, gen_len=4)
    sampled_tokens = actions.squeeze(0).tolist()
    raw_reward = reward_function(sampled_tokens)

    policy_logits = policy_model(full_seq)[:, :-1, :]
    ref_logits = ref_model(full_seq)[:, :-1, :]

    gen_policy_logits = policy_logits[:, prompt.shape[1]-1:, :]
    gen_ref_logits = ref_logits[:, prompt.shape[1]-1:, :]

    p_policy = F.softmax(gen_policy_logits, dim=-1)
    p_ref = F.softmax(gen_policy_logits, dim=-1)

    kl_div = torch.sum(p_policy * (torch.log(p_policy + 1e-8) - torch.log(p_ref + 1e-8)), dim=-1).mean()

    net_reward = raw_reward - (beta_kl * kl_div.item())

    loss = -action_log_probs.sum() * net_reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == 1:
        words = " ".join([VOCAB[t] for t in sampled_tokens])
        print(f"Epoch: {epoch:3d} | Sampled: {words:26s} | "
              f"Reward: {raw_reward:4.1f} | KL: {kl_div.item():.4f} | Loss: {loss.item():.4f}")

print("\n--- Final generations after RLHF ---")
policy_model.eval()
with torch.no_grad():
    for _ in range(3):
        full_seq, _, _ = generate_sequence(policy_model, prompt, gen_len=4)
        tokens = [VOCAB[t.item()] for t in full_seq[0]]
        print("Generated: ", " ".join(tokens[1:]))


