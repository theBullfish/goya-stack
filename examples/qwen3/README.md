# Qwen3-30B-A3B on one Goya card

A 30-billion-parameter mixture-of-experts language model generating text on a single HL-100.
Every matrix multiply runs as int8 on the card's MME. Norms, rotary embedding, the softmax
pieces and SwiGLU run as f32 kernels on the TPC cores. The host handles tokenising, routing and
bookkeeping in numpy.

It is slow, and it is real. The reference run took 158.5 s for 4 tokens. The first token took
104 s, which covers the 5-token prompt plus one-time preparation of the weights it touched.
It is a working end-to-end demonstration of the card, not a fast chatbot.

## 1. Get the model (32.5 GB)

This exact file, from a pinned revision. The reference check depends on its exact bytes:

```bash
curl -L -o Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf \
  https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/eea7b2be5805a5f151f8847ede8e5f9a9284bf77/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf
echo "47514861adaa7ae7f32818fe54595c7cbed2753417ba78df53054fcf7b9dcd88  Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf" | sha256sum -c -
```

The model is Qwen3-30B-A3B-Instruct-2507 by Alibaba's Qwen team, Apache-2.0, quantised to Q8_0
by Unsloth.

## 2. Run the reference check

Needs `setup.sh` to have finished, and numpy (`sudo apt install python3-numpy`):

```bash
sudo python3 generate.py --model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf --check
```

Expected:

```
prompt 'The capital of France is' -> 5 tokens -> 'The capital of France is'
  [0] ... s  -> 12095  ' Paris'
  [1] ... s  -> 13     '.'
  [2] ... s  -> 576    ' The'
  [3] ... s  -> 6722   ' capital'

=== OUTPUT ===
The capital of France is Paris. The capital
CHECK PASS: the 4 tokens match the reference run exactly.
```

## 3. Your own prompt

```bash
sudo python3 generate.py --model Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf --prompt "Write a haiku about" --tokens 16
```

| option | meaning |
|---|---|
| `--card 0000:03:00.0` | use a specific card (default: the first free one) |
| `--tokens N` | how many new tokens to generate (default 8) |
| `--expert-cache-gb G` | host RAM for quantised expert weights (default 8). Lower it on a smaller machine; it only changes speed, not the answer. |

Generation is greedy (always the most likely token), so the same prompt gives the same text.
The prompt goes in raw, with no chat template.

## Files

- `generate.py`: the pipeline. It is the script from the reference run, with two changes: it
  reads the model with `gguf_min.py`, and it bounds host memory use.
- `gguf_min.py`: a small GGUF reader, numpy only. We checked it against the reader used in the
  reference run on all 579 tensors of this model: same metadata, vocabulary, tensor locations
  and decoded weights.
