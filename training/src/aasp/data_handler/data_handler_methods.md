# AASPDataHandler — Method Specifications

This document describes the purpose and required behavior of each public method
in the `AASPDataHandler` class.  
Implementation lives separately in `data_handler.py`.

## Core Concept

`AASPDataHandler` converts the **raw list-of-dicts records** loaded from a `.pkl`
file → cleaned, filtered, encoded numpy arrays suitable for `AASPDataset` and PyTorch.

`fields` in the config represent **input features only**.  
`score` is always preserved separately as the target.

---

## Methods

### `load_pickle(path=None)`
Load the full dataset from a `.pkl` file and return records as a Python list of dicts.  
This function **does not filter or select fields** — raw load only.

---

### `select_fields(records, fields=None)`
Return new records that keep only the specified **input fields** plus `"score"`.  
`fields` defaults to the values in the `AASPConfig`.

---

### `filter(records, *, scoresets=None, biotypes=None, max_rows=None)`
Return a subset of records based on optional criteria, useful before splitting.  
This is purely a convenience filter.

---

### `split(records, *, val_frac=None, test_frac=None, seed=None, group_by=None)`
Split into `(train, val, test)`.  
If `group_by` is provided (e.g. `"scoreset"`), groups must not be split across partitions.

---

### `fit_vocab(records, key, add_unk=True, unk_token="<UNK>")`
Build a string → integer ID mapping for categorical fields.  
Should be called only on TRAIN to avoid leakage.

---

### `encode(records, key, vocab, unk_token="<UNK>")`
Convert category strings into integer ID numpy array based on a given vocab.

---

### `one_hot(ids, num_classes)`
Convert integer category IDs to one-hot vectors.  
Primarily for baselines and debugging.

---

### `get_numeric(records, keys)`
Extract numeric columns into a `[N, K]` numpy array.

---

### `get_target(records, key="score")`
Extract the target vector into shape `[N, 1]`.  
Score is always available even if not in `fields`.

---

### `get_embedding(records, key, *, pad_to=None, truncate_to=None)`
Extract embedding vectors (e.g. `ref_embedding`, `alt_embedding`) into a dense 2-D array.  
Must guarantee fixed dimension either by verifying constant length or 
using pad/truncate.

---

### `fuse_embeddings(ref_emb, alt_emb, how="concat")`
Combine reference and alternate embedding matrices. Options:

* `"concat"` → `[ref, alt]`
* `"diff"` → `alt - ref`
* `"sum"` → `alt + ref`

---

### `get_sequence(records, key="sequence")`
Return list of sequence strings.

---

### `tokenize_sequence(seqs, alphabet=20AA, unk_token="X")`
Map characters in sequences to integer IDs.  
(Useful for simple baselines — PLMs have their own tokenizers.)

---

### `fit_scaler(X)`
Compute mean/std for numeric features for standardization.

---

### `apply_scaler(X, stats)`
Apply standardization using stats from `fit_scaler()`.

---

### `cache_arrays(name, **arrays)`
Write named numpy arrays to disk (e.g. `.npz`) for caching.

---

### `load_cached(name)`
Load cached arrays previously written by `cache_arrays`.

---

### `summary(records)`
Return a compact printable summary describing the dataset: top categories, counts, missing data, etc.

---
