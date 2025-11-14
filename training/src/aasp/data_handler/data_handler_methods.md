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

### `encode(records, key, vocab, unk_token="<UNK>")`
Convert category strings into integer ID numpy array based on a given vocab.

---

### `one_hot(ids, num_classes)`
Convert integer category IDs to one-hot vectors.  
Primarily for baselines and debugging.

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

### `summary(records)`
Return a compact printable summary describing the dataset: top categories, counts, missing data, etc.

---
