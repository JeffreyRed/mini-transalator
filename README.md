# mini-translator

> English→Spanish neural machine translation from scratch.
> Step 7 of the mini-LLM series.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

## Series

| Step | Repository | What it builds |
|------|-----------|----------------|
| 1 | [mini-embedding](https://github.com/JeffreyRed/mini-embedding) | Word vectors — Skip-gram Word2Vec |
| 2 | [mini-self-attention](https://github.com/JeffreyRed/mini-self-attention) | Multi-head self-attention |
| 3 | [mini-transformer](https://github.com/JeffreyRed/mini-transformer) | Positional encoding + causal decoder |
| 4 | [mini-gpt](https://github.com/JeffreyRed/mini-gpt) | Real corpus, overfitting, evaluation |
| 5 | [mini-chat](https://github.com/JeffreyRed/mini-chat) | Instruction format, loss masking |
| 6 | [mini-cross-attention](https://github.com/JeffreyRed/mini-cross-attention) | Cross-attention module, alignment |
| **7** | **mini-translator** ← you are here | English→Spanish NMT, BLEU, soft alignment |

---

## What this adds over mini-cross-attention

| Feature | mini-cross-attention | mini-translator |
|---|---|---|
| Task | digit reversal (synthetic) | English→Spanish (real language) |
| Vocabularies | shared (digits only) | **separate** — English encoder, Spanish decoder |
| Alignment pattern | anti-diagonal (reversal) | **soft diagonal** (similar word order) |
| Evaluation metric | token accuracy | **BLEU score** |
| Corpus | generated programmatically | 140 curated sentence pairs |
| Layers | 2 encoder + 2 decoder | **4 encoder + 4 decoder** |

---

## What you will see

### Alignment — soft diagonal

Unlike the clean anti-diagonal from digit reversal, English→Spanish produces
a **soft diagonal from top-left to bottom-right**, reflecting similar word order:

```
Translating: "the cat sits on the mat"
             "el gato se sienta en la alfombra"

         the   cat  sits   on   the   mat
  el    [0.82, 0.08, 0.04, 0.02, 0.02, 0.02]   ← "el" looked at "the"
  gato  [0.07, 0.79, 0.06, 0.04, 0.02, 0.02]   ← "gato" looked at "cat"
  se    [0.03, 0.05, 0.74, 0.08, 0.05, 0.05]   ← "se" looked at "sits"
  sienta[0.02, 0.04, 0.72, 0.09, 0.07, 0.06]   ← "sienta" looked at "sits"
  en    [0.02, 0.02, 0.06, 0.82, 0.04, 0.04]   ← "en" looked at "on"
  la    [0.02, 0.02, 0.03, 0.04, 0.81, 0.08]   ← "la" looked at "the"
  alfombra[0.01, 0.02, 0.03, 0.02, 0.05, 0.87] ← "alfombra" looked at "mat"
```

The bright cells form a diagonal — semantically corresponding words align.
The weights are not 1.0 because Spanish grammar sometimes requires looking
at surrounding context (articles, verb conjugation depends on subject).

### BLEU score

BLEU measures overlap between the generated translation and the reference:

```
BLEU = 0.0    no shared words at all
BLEU = 0.3    reasonable quality
BLEU = 0.5+   good quality on a clean controlled corpus like this one
BLEU = 1.0    exact match (never happens on real data)
```

---

## Architecture

```
English input  [the, cat, sits, on, the, mat]
      │
      ▼
┌──────────────────────────────────────┐
│  src_embedding  (English vocab → D)  │  separate from Spanish
│  PositionalEncoding                  │
│  EncoderBlock × 4  (self-attention)  │
└──────────────────────────────────────┘
      │  encoder_output  (batch, src_len, D)
      │
      │    Spanish input  [BOS, el, gato, se, sienta, ...]
      │          │
      ▼          ▼
┌──────────────────────────────────────┐
│  tgt_embedding  (Spanish vocab → D)  │  separate from English
│  PositionalEncoding                  │
│  DecoderBlock × 4                    │
│    1. Causal self-attention          │  Spanish attends to past Spanish
│    2. Cross-attention ←──────────────┤  Spanish queries English encoder
│    3. FeedForward                    │
└──────────────────────────────────────┘
      │
      ▼
  Linear → Spanish vocab logits
  → [el, gato, se, sienta, en, la, alfombra, EOS]
```

---

## Project structure

```
mini-translator/
│
├── data/
│   └── corpus.txt              # 140 English\tSpanish sentence pairs
│
├── src/
│   ├── tokenizer.py            # separate English + Spanish vocabularies
│   ├── attention.py            # self-attention + cross-attention
│   ├── model.py                # MiniTranslator (4+4 encoder-decoder)
│   ├── dataset.py              # TranslationDataset
│   ├── train.py                # training loop + BLEU + early stopping
│   ├── utils.py                # translate(), interactive loop
│   └── visualize.py            # alignment heatmaps, BLEU curves, vocab plot
│
├── outputs/
├── main.py
├── environment.yml
├── requirements.txt
├── THEORY.md
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/your-username/mini-translator.git
cd mini-translator
conda env create -f environment.yml
conda activate mini-translator
python main.py
```

**Training time:** 5–10 minutes on CPU.

---

## Configuration

| Parameter | Default | Notes |
|---|---|---|
| `EMB_DIM` | `128` | Larger than previous steps — real vocabulary needs capacity |
| `N_HEADS` | `4` | head_dim = 32 |
| `N_LAYERS` | `4` | 4 encoder + 4 decoder blocks |
| `FF_DIM` | `256` | 2 × EMB_DIM |
| `EPOCHS` | `80` | Early stopping typically triggers before this |
| `LR` | `3e-3` | Peak cosine LR |
| `PATIENCE` | `15` | Early stopping patience (on BLEU) |

---

## Outputs

| File | Description |
|---|---|
| `translator.pt` | Saved model + config + best BLEU |
| `vocab_comparison.png` | English vs Spanish vocabulary side by side |
| `training.png` | BLEU + perplexity curves (train vs val) |
| `alignment_0.png` | Soft diagonal alignment — "the cat sits on the mat" |
| `alignment_1.png` | "imagination is more important than knowledge" |
| `alignment_2.png` | "plants need sunlight to grow" |
| `alignment_3.png` | "the brain processes information quickly" |
| `alignment_4.png` | "learning never stops" |
| `alignment_layers.png` | Layer 1→4 alignment for first sentence — sharpening visible |

---

## Deep dive

See [`THEORY.md`](./THEORY.md) for:

- Why separate vocabularies and how cross-attention bridges them
- BLEU score — the formula and how to interpret it
- Why the alignment is diagonal (not anti-diagonal like reversal)
- Where the alignment is fuzzy and why
- Layer-by-layer sharpening of alignment
- The full training data flow
- How this connects to production NMT systems like Google Translate

---

## References

- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Bahdanau et al. (2015) — [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Papineni et al. (2002) — [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)

---

## License

MIT
# mini-transalator
