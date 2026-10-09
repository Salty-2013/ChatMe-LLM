import numpy as np ##GOAT##
import pandas as pd
import torch as tch

##LLM MODEL NAME: ChatMe##

epochs = 10000
learning_rate = 0.001
block_size = 256
n_embd = 300
n_heads = 12
heads_dim = n_embd // n_heads
vocab_size = 0
batch_size = 32


data = pd.read_csv("MAIN AI/Dataset.csv")
text = "\n".join(str(x) for x in data["text"])
print(len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
STRtoI = {ch : x for x, ch in enumerate(chars)}
ItoSTR = {x : ch for x, ch in enumerate(chars)}
print("Vocabulary size:", vocab_size)

encode = lambda y : [STRtoI[x] for x in y]
decode = lambda a : ''.join([ItoSTR[b] for b in a])
data = encode(text)

class ChatMe(tch.nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = tch.nn.LayerNorm(n_embd)
        self.ln2 = tch.nn.LayerNorm(n_embd)

        self.w_token = tch.nn.Parameter(tch.randn(vocab_size, n_embd) * 0.02)
        self.w_pos = tch.nn.Parameter(tch.randn(block_size, n_embd) * 0.02)
        self.w_Q = tch.nn.Parameter(tch.randn(n_embd, n_embd) * 0.02)
        self.w_K = tch.nn.Parameter(tch.randn(n_embd, n_embd) * 0.02)
        self.w_V = tch.nn.Parameter(tch.randn(n_embd, n_embd) * 0.02)
        self.w_proj = tch.nn.Parameter(tch.randn(n_embd, n_embd) * 0.02)
        ff_hidden = 4 * n_embd
        self.w_ff1 = tch.nn.Parameter(tch.randn(n_embd, ff_hidden) * 0.02)
        self.b_ff1 = tch.nn.Parameter(tch.zeros(ff_hidden))
        self.w_ff2 = tch.nn.Parameter(tch.randn(ff_hidden, n_embd) * 0.02)
        self.b_ff2 = tch.nn.Parameter(tch.zeros(n_embd))
        self.w_lm = tch.nn.Parameter(tch.randn(n_embd, vocab_size) * 0.02)
        self.b_lm = tch.nn.Parameter(tch.zeros(vocab_size))

    def forward(self, Q):
        B, T = Q.shape

        token_emb = self.w_token[Q]
        pos_emb = self.w_pos[:T]

        CE = token_emb + pos_emb #Combined Embedding#
        CE = self.ln1(CE)

        q = CE @ self.w_Q
        k = CE @ self.w_K
        v = CE @ self.w_V

        Q_heads = q.reshape(B, T, n_heads, heads_dim).transpose(1, 2)
        K_heads = k.reshape(B, T, n_heads, heads_dim).transpose(1, 2)
        V_heads = v.reshape(B, T, n_heads, heads_dim).transpose(1, 2)

        atn_scores = Q_heads @ K_heads.transpose(-2, -1)
        atn_scores /= heads_dim ** 0.5

        mask = tch.tril(tch.ones(T, T, device=Q.device, dtype=tch.bool))
        atn_scores = atn_scores.masked_fill(~mask, -1e9)

        atn_probs = tch.softmax(atn_scores, dim = - 1)
        atn_output = atn_probs @ V_heads
        atn_output = atn_output.transpose(1, 2)
        atn_output = atn_output.reshape(B, T, n_embd)
        atn_projected = atn_output @ self.w_proj

        x = CE + atn_projected
        x = self.ln2(x)

        ff1 = x @ self.w_ff1 + self.b_ff1
        ff1 = tch.relu(ff1)
        ff2 = ff1 @ self.w_ff2 + self.b_ff2

        x = x + ff2

        logits = x @ self.w_lm + self.b_lm
        return logits

model = ChatMe()
model.load_state_dict(tch.load("MAIN AI/LLM MEMORY/ChatMe.pt"))
device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

@tch.no_grad()
def generate(start_str, max_tokens=256, temperature=0.7):
    context = encode(start_str)

    for _ in range(max_tokens):
        x_cond = context[-block_size:]

        X = tch.tensor(
            [x_cond],
            dtype=tch.long,
            device=device
        )
        logits = model(X)
        logits = logits[0, -1, :] / max(temperature, 1e-5)
        probs = tch.softmax(logits, dim=0)
        ix = tch.multinomial(probs, num_samples=1)
        context.append(ix.item())

    return decode(context)

optimizer = tch.optim.AdamW(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    ix = tch.randint(0, len(data) - block_size - 1, (batch_size,))
    X = tch.stack([tch.tensor(data[i:i + block_size], dtype=tch.long)for i in ix]).to(device)
    Y = tch.stack([tch.tensor(data[i + 1:i + block_size + 1], dtype=tch.long)for i in ix]).to(device)
    logits = model(X)

    loss = tch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), Y.reshape(-1))
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

tch.save(model.state_dict(), "MAIN AI/LLM MEMORY/ChatMe.pt")
print("LLM BRAIN has been trained and saved")
model.eval()

while True:
    Ask = input("\nAsk ChatMe: ")
    print(f"ChatMe: {generate(Ask, max_tokens=256, temperature = 0.7)}")
    if Ask.lower() == "exit":
        print("Please go and never come front of my face again. I will not answer you anymore. Goodbye.")
        break