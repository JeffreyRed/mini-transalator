# Theory & Code Walkthrough — mini-translator

> Step 7 of the mini-LLM series. Prerequisite: [mini-cross-attention](https://github.com/JeffreyRed/mini-cross-attention).

---

## Table of Contents

1. [What this step adds](#1-what-this-step-adds)
2. [Separate vocabularies — why and how](#2-separate-vocabularies--why-and-how)
3. [Why the alignment is diagonal, not anti-diagonal](#3-why-the-alignment-is-diagonal-not-anti-diagonal)
4. [Where the alignment is fuzzy and why](#4-where-the-alignment-is-fuzzy-and-why)
5. [Layer-by-layer sharpening](#5-layer-by-layer-sharpening)
6. [BLEU score — the translation metric](#6-bleu-score--the-translation-metric)
7. [How cross-attention bridges two embedding spaces](#7-how-cross-attention-bridges-two-embedding-spaces)
8. [Code walkthrough](#8-code-walkthrough)
9. [Full data flow](#9-full-data-flow)
10. [Connection to production NMT](#10-connection-to-production-nmt)

---

## 1. What this step adds

`mini-cross-attention` demonstrated the mechanism on a synthetic task (digit reversal)
where the alignment was perfectly interpretable (anti-diagonal) but artificial.

`mini-translator` applies the same mechanism to real language. The new pieces are:

| Addition | Why it matters |
|---|---|
| Real bilingual corpus (140 pairs) | Model must learn actual word correspondences |
| Separate English/Spanish vocabularies | More realistic; shows cross-attention bridging two spaces |
| BLEU metric | Standard measure of translation quality |
| 4+4 encoder-decoder layers | Real vocabulary needs more representational capacity |
| Soft diagonal alignment | Shows English≈Spanish word order and fuzzy language correspondences |

---

## 2. Separate vocabularies — why and how

`mini-cross-attention` used a shared vocabulary of 13 digit tokens. That worked
because source and target were the same symbols. In translation they are not.

English and Spanish have different words, different morphology, and different
vocabulary sizes. Each language gets its own embedding matrix:

```
src_embedding: (English_vocab_size × emb_dim)
tgt_embedding: (Spanish_vocab_size × emb_dim)
```

Both have the same `emb_dim` so cross-attention dimensions match:

```
Encoder output:  (batch, src_len, emb_dim)   ← English vectors
Decoder state:   (batch, tgt_len, emb_dim)   ← Spanish vectors

Cross-attention:
  Q = W_Q(decoder_state)     → (batch, heads, tgt_len, head_dim)   from Spanish
  K = W_K(encoder_output)    → (batch, heads, src_len, head_dim)   from English
  V = W_V(encoder_output)    → (batch, heads, src_len, head_dim)   from English

  Q @ K^T  →  (batch, heads, tgt_len, src_len)   alignment matrix
```

The cross-attention weight matrices W_Q, W_K, W_V are the bridge. They
learn to project Spanish decoder states and English encoder states into a
shared query-key space where semantically related words have high dot products.

This is why cross-attention can work even when the two embedding spaces were
initialised independently. The projection matrices align the spaces during training.

---

## 3. Why the alignment is diagonal, not anti-diagonal

In `mini-cross-attention` the alignment was anti-diagonal because:
- Output is generated left to right
- Task requires reading source right to left (reversal)
- → bright cells run from top-right to bottom-left

In `mini-translator` the alignment is **diagonal from top-left to bottom-right** because:
- Output is generated left to right
- English and Spanish have **similar word order** (both roughly SVO: Subject-Verb-Object)
- → to generate Spanish word `i`, the model typically looks at English word `i` or nearby

```
English:  the    cat    sits    on    the    mat
          ↕      ↕      ↕       ↕     ↕      ↕
Spanish:  el    gato    se    sienta  en     la    alfombra
```

Most words map nearly one-to-one in order, so the alignment matrix has
bright cells running diagonally from top-left to bottom-right.

This diagonal pattern is the visual signature of languages with similar
word order. If you translated English to Japanese (which has verb-final order),
the alignment would look quite different — the verb would appear at the end
of the Japanese output but somewhere in the middle of the English input,
creating a non-diagonal pattern.

---

## 4. Where the alignment is fuzzy and why

The alignment is never perfectly sharp (1.0 in one cell, 0.0 everywhere else).
Several linguistic phenomena cause spreading:

**Articles and determiners:**
"the cat" → "el gato". The Spanish article "el" attends mainly to "the" but
also partly to "cat" because which article to use (el/la/los/las) depends on
the gender of the noun. The model must look ahead.

**Verb morphology:**
"sits" → "se sienta". Spanish verb conjugation encodes person, number, tense.
To conjugate correctly, the decoder must attend to the subject ("cat" → 3rd
person singular) as well as to "sits" itself.

**Compound translations:**
"sits" requires two Spanish words: "se sienta" (reflexive particle + verb).
Both Spanish tokens must attend to the same English token, so two rows of
the alignment matrix will both peak at the same column.

**Function words:**
Prepositions like "on" → "en" typically align cleanly, but their Spanish
equivalents can depend on the following noun, causing some spreading.

This fuzziness is not a failure. It reflects genuine linguistic complexity.
Human translators also need context beyond the single word they are translating.

---

## 5. Layer-by-layer sharpening

`outputs/alignment_layers.png` shows the cross-attention averaged across
heads for each decoder layer (1 through 4).

The typical pattern:

```
Layer 1:  diffuse, roughly uniform rows
           The model is still building representations; not yet certain
           which source word each target word corresponds to.

Layer 2:  mild diagonal pattern begins to emerge
           Some correspondences are becoming clearer.

Layer 3:  stronger diagonal, some cells clearly dominant

Layer 4:  sharpest alignment — closest to what we showed in the example above
           Later layers have more refined representations to work with.
```

This sharpening happens because each decoder layer receives the output of
the previous layer as input. By layer 4, the decoder state for "gato" already
encodes substantial information about what kind of word it is (an animal noun),
which makes the query for cross-attention more specific and the resulting
alignment more focused.

This is the "refinement" behaviour we first observed in `mini-transformer`
with `compare_layers()`, now visible in the alignment matrix.

---

## 6. BLEU score — the translation metric

BLEU (Bilingual Evaluation Understudy) is the standard automatic metric
for translation quality. It measures how much the generated translation
overlaps with a reference translation using n-gram precision.

### The formula

For a hypothesis (generated translation) and a reference:

```
BLEU = BP × exp( Σ_n  w_n × log(p_n) )

where:
  p_n   = n-gram precision  (fraction of hypothesis n-grams that appear in reference)
  w_n   = weight for each n-gram order  (usually 1/N for N orders)
  BP    = brevity penalty  = min(1,  exp(1 - |reference| / |hypothesis|))
```

The brevity penalty discourages very short translations that achieve high
precision by saying almost nothing.

### Interpretation

```
BLEU = 0.0   completely wrong — no n-gram overlap
BLEU = 0.1   poor — some words correct but mostly wrong
BLEU = 0.3   reasonable — the meaning is roughly conveyed
BLEU = 0.5   good — most words correct on a clean controlled corpus
BLEU = 0.7+  very good — typically requires much larger models and data
BLEU = 1.0   exact match — almost never happens even for human translators
             (paraphrases exist; reference is just one valid translation)
```

On this corpus (140 clean sentence pairs, controlled vocabulary):
- BLEU > 0.5 is achievable after sufficient training
- Exact matches will be common because the corpus has no paraphrases

### Why BLEU is imperfect

BLEU is a proxy metric. It penalises valid paraphrases that differ from the
reference. "the cat is sitting" and "the cat sits" are both valid translations
of "el gato se sienta" but BLEU would not give full credit unless the reference
matches. For this series, BLEU is a good enough training signal and progress
indicator.

---

## 7. How cross-attention bridges two embedding spaces

This is the deepest question in this project: how can the decoder, which
has only ever seen Spanish vectors, successfully query the encoder, which
has only ever seen English vectors?

The answer: **the projection matrices W_Q, W_K, W_V learn to create a
shared intermediate space.**

At initialisation, English and Spanish embedding spaces are random and
completely unrelated. After training:

```
W_Q projects Spanish decoder states into "query space"
W_K projects English encoder states into "key space"

For cross-attention to work, the dot product Q · K^T must be large
when the Spanish and English words are translations of each other
and small otherwise.

So W_Q and W_K jointly learn a transformation such that:
  W_Q("gato" state) · W_K("cat" state)^T   →   high score
  W_Q("gato" state) · W_K("rain" state)^T  →   low score
```

Neither W_Q nor W_K can do this alone. They must co-adapt — the query
projection and key projection both adjust so that their outputs are
comparable. This co-adaptation is what "learning to align" means.

The value projection W_V then determines what information is actually
passed to the decoder once the alignment is established. It learns to
extract the aspects of the English representation that are most useful
for generating the Spanish translation.

---

## 8. Code walkthrough

### `tokenizer.py`

**`Vocabulary`** is a standalone class — one per language.

**`TranslationTokenizer`** holds two `Vocabulary` instances:

```python
all_src = [w for s, _ in pairs for w in s]
all_tgt = [w for _, t in pairs for w in t]
self.src_vocab = Vocabulary(all_src, min_freq)
self.tgt_vocab = Vocabulary(all_tgt, min_freq)
```

The source is encoded **without** BOS/EOS — the encoder sees raw tokens.
The target is encoded **with** BOS/EOS — the decoder uses teacher forcing
starting from BOS.

```python
def encode_pair(self, src, tgt):
    return (
        self.src_vocab.encode(src, add_special=False),  # no BOS/EOS
        self.tgt_vocab.encode(tgt, add_special=True),   # BOS + words + EOS
    )
```

---

### `model.py`

**Separate embeddings:**

```python
self.src_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=src_pad_idx)
self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=tgt_pad_idx)
self.src_pe  = PositionalEncoding(emb_dim, max_len, dropout)
self.tgt_pe  = PositionalEncoding(emb_dim, max_len, dropout)
```

Each language has its own embedding matrix and positional encoding.
They share `emb_dim` so cross-attention dimensions are compatible.

**`encode()` is called once; `decode()` uses the cached result:**

```python
def forward(self, src, tgt):
    enc_out, src_mask    = self.encode(src)     # English → encoder output
    dec_out, all_cross_w = self.decode(tgt, enc_out, src_mask)
    return self.head(dec_out), all_cross_w
```

At inference (`translate()`):

```python
enc_out, src_mask = self.encode(src)   # run ONCE
for each decoding step:
    dec_out, cws = self.decode(tgt_so_far, enc_out, src_mask)
    next_token   = head(dec_out)[0, -1, :].argmax()
```

The encoder is never re-run. Its output is reused at every decoding step.

---

### `train.py`

**`simple_bleu()`** computes 2-gram BLEU for one sentence pair:

```python
for n in range(1, max_n + 1):
    hyp_ngrams = Counter(tuple(hypothesis[i:i+n]) ...)
    ref_ngrams = Counter(tuple(reference[i:i+n]) ...)
    matches    = sum((hyp_ngrams & ref_ngrams).values())
    scores.append(matches / sum(hyp_ngrams.values()))

bp   = min(1.0, exp(1 - len(reference) / len(hypothesis)))
bleu = bp * exp(mean(log(s) for s in scores))
```

The `&` operation on two Counters gives the minimum count for each key —
this handles repeated n-grams correctly without overcounting.

**Early stopping is on BLEU** (not loss) because BLEU directly measures
translation quality. A model can have lower loss without producing better
translations if it memorises rare word distributions.

---

## 9. Full data flow

```
corpus.txt
  "the cat sits on the mat\tel gato se sienta en la alfombra"
        │
        ▼  TranslationTokenizer
  English: ["the", "cat", "sits", "on", "the", "mat"]
  Spanish: ["el", "gato", "se", "sienta", "en", "la", "alfombra"]
        │
        ▼  encode_pair()
  src_ids:  [7, 12, 43, 21, 7, 35]          (no BOS/EOS)
  tgt_ids:  [1, 8, 17, 31, 44, 11, 19, 40, 2]  (BOS...EOS)
        │
        ▼  TranslationDataset
  src:      [7, 12, 43, 21, 7, 35]
  dec_in:   [1, 8, 17, 31, 44, 11, 19, 40]   (tgt[:-1])
  dec_tgt:  [8, 17, 31, 44, 11, 19, 40, 2]   (tgt[1:])
        │
        ▼  MiniTranslator.forward(src, dec_in)

  ENCODER:
  src_emb([7,12,43,21,7,35])  →  (1, 6, 128)   English vectors
  + positional encoding
  → 4 × EncoderBlock (self-attention, no mask)
  → encoder_output  (1, 6, 128)

  DECODER:
  tgt_emb([1,8,17,31,44,11,19,40])  →  (1, 8, 128)   Spanish vectors
  + positional encoding
  → 4 × DecoderBlock:
      causal self-attention  (Spanish → past Spanish, causal mask)
      cross-attention        Q from Spanish, K/V from encoder_output
        → cross_w  (1, 4, 8, 6)  ← alignment matrix: 8 tgt × 6 src
      feedforward
  → head  →  logits  (1, 8, tgt_vocab_size)

        ▼  CrossEntropyLoss(logits, dec_tgt)
  loss at position 0: model saw BOS, must predict "el"
  loss at position 1: model saw BOS+el, must predict "gato"
  ...
  loss averaged over 8 positions

        ▼  backward + Adam step
  W_Q, W_K, W_V in cross-attention nudged so that:
    "gato" query aligns with "cat" key  →  higher attention score
    "gato" query misaligns with "rain" key  →  lower score
```

---

## 10. Connection to production NMT

Modern neural machine translation systems like Google Translate, DeepL,
and Meta's NLLB use the same architecture — scaled up enormously:

| Component | mini-translator | Google Translate (approx) |
|---|---|---|
| Architecture | encoder-decoder transformer | encoder-decoder transformer |
| Encoder layers | 4 | 6–24 |
| Decoder layers | 4 | 6–24 |
| emb_dim | 128 | 512–1024 |
| Corpus size | 140 sentence pairs | billions of sentence pairs |
| Vocabulary | word-level (~300 tokens each) | subword BPE (~32,000 tokens) |
| Training | minutes on CPU | weeks on thousands of GPUs |
| BLEU (newstest) | N/A (tiny corpus) | ~30–40 on standard benchmarks |

The key differences are scale and tokenization. Production systems use
**subword tokenization (BPE or SentencePiece)** which splits rare words
into smaller pieces, allowing the model to handle words not seen in training.
Everything else — the encoder-decoder structure, cross-attention alignment,
teacher forcing, BLEU evaluation — is identical in principle.

---

*This completes the mini-LLM series core sequence. The full progression from
word vectors to a neural machine translator demonstrates every major concept
in modern NLP: embeddings, self-attention, positional encoding, causal
language modelling, instruction fine-tuning, cross-attention, and alignment.*
