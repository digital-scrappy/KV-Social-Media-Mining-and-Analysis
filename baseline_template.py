#!/usr/bin/env python
# coding: utf-8

# # Unimodal hateful-meme baseline (template)
# One notebook per experiment. Copy this file, change ONLY the config cell, run top to bottom.
# Shared logic lives in `common.py` so all four runs use identical training/eval code.

# In[1]:


# ---------------- CONFIG (the only cell that differs per experiment) ----------------
EXP_NAME   = 'text_A'          # image_A | image_B | text_A | text_B
MODALITY   = 'text'            # 'image' or 'text'
MODEL_NAME = 'google/embeddinggemma-2'   # exact HF checkpoint
IMG_KIND   = 'clip'             # image only: 'clip' (get_image_features) or 'cls' (CLS token, e.g. DINOv2)
POOLING    = 'mean'             # text only: 'mean' or 'cls'
MAX_LEN    = 8192                 # text only: max tokens (truncation)

DATA_DIR   = 'data/'                # folder containing train.jsonl, dev.jsonl, test.jsonl and img/
OUT_DIR    = f'outputs/{EXP_NAME}'
SEEDS      = [0, 1, 2]          # same for ALL experiments
GRID       = {'lr': [1e-4, 1e-3, 1e-2], 'wd': [0, 1e-4, 1e-2], 'bs': [64, 256]}  # same for ALL experiments
EPOCHS     = 100
USE_TEST   = False              # flip to True ONLY when the course protocol allows test evaluation


# In[2]:


import os, json, numpy as np, pandas as pd, torch
import common as C
os.makedirs(OUT_DIR, exist_ok=True)
print('device:', C.DEVICE)
C.describe_splits(DATA_DIR)


# In[3]:


# ---------------- 1. Extract + cache embeddings (run once) ----------------
train, dev, test = (C.load_split(DATA_DIR, n) for n in ['train', 'dev', 'test'])

def embed(rows):
    if MODALITY == 'image':
        return C.extract_image_embeddings(rows, DATA_DIR, MODEL_NAME, IMG_KIND)
    return C.extract_text_embeddings(rows, MODEL_NAME, POOLING, MAX_LEN)

emb = {}
for name, rows in [('train', train), ('dev', dev), ('test', test)]:
    path = f'{OUT_DIR}/emb_{name}.npy'
    if os.path.exists(path):
        emb[name] = np.load(path)
    else:
        emb[name] = embed(rows)
        np.save(path, emb[name])
    print(name, emb[name].shape)
EMB_DIM = emb['train'].shape[1]


# In[ ]:


# ---------------- 2. Prepare features / labels ----------------
ytr, yva = C.get_labels(train), C.get_labels(dev)
yte = C.get_labels(test) if 'label' in test[0] else None
Xtr, Xva, Xte = C.standardize(emb['train'], emb['dev'], emb['test'])


# In[ ]:


# ---------------- 3. Hyperparameter search (validation AUROC, seed = SEEDS[0]) ----------------
search = C.grid_search(Xtr, ytr, Xva, yva, GRID, seed=SEEDS[0], epochs=EPOCHS)
pd.DataFrame(search).to_csv(f'{OUT_DIR}/grid_search.csv', index=False)
best = search[0]
print('best:', best)
pd.DataFrame(search).head(10)


# In[ ]:


# ---------------- 4. Final runs over all seeds ----------------
per_seed, first = C.final_runs(Xtr, ytr, Xva, yva, Xte if USE_TEST else None, yte if USE_TEST else None, best, SEEDS, epochs=EPOCHS)
df = pd.DataFrame(per_seed); display(df)
print('val  AUROC mean/std:', C.summarize(per_seed, 'val_auroc'))
if USE_TEST:
    print('test AUROC mean/std:', C.summarize(per_seed, 'test_auroc'))


# In[ ]:


# ---------------- 5. Save everything needed by the report / error analysis ----------------
torch.save(first['state'], f'{OUT_DIR}/classifier_seed{SEEDS[0]}.pt')
pd.DataFrame({'id': [r['id'] for r in dev], 'label': yva, 'score': first['val_scores']}).to_csv(f'{OUT_DIR}/val_scores.csv', index=False)
if USE_TEST and first['test_scores'] is not None:
    pd.DataFrame({'id': [r['id'] for r in test], 'score': first['test_scores']}).to_csv(f'{OUT_DIR}/test_scores.csv', index=False)
json.dump({'exp': EXP_NAME, 'model': MODEL_NAME, 'modality': MODALITY, 'emb_dim': int(EMB_DIM),
           'img_kind': IMG_KIND if MODALITY == 'image' else None,
           'pooling': POOLING if MODALITY == 'text' else None, 'max_len': MAX_LEN if MODALITY == 'text' else None,
           'optimizer': 'Adam', 'epochs_max': EPOCHS, 'seeds': SEEDS, 'grid': GRID, 'best_hparams': best,
           'per_seed': per_seed}, open(f'{OUT_DIR}/summary.json', 'w'), indent=2)
print('saved to', OUT_DIR)


# ## Error analysis (run after BOTH partners have finished)
# Load `outputs/<best_image>/val_scores.csv` and `outputs/<best_text>/val_scores.csv`, merge on `id`, then pick the 5 dev examples with the largest disagreement at threshold 0.5.

# In[ ]:


# BEST_IMG, BEST_TXT = 'image_A', 'text_A'   # fill in after comparing validation AUROC
# a = pd.read_csv(f'outputs/{BEST_IMG}/val_scores.csv').rename(columns={'score': 'img_score'})
# b = pd.read_csv(f'outputs/{BEST_TXT}/val_scores.csv').rename(columns={'score': 'txt_score'})[['id', 'txt_score']]
# m = a.merge(b, on='id')
# m['disagree'] = (m.img_score >= 0.5) != (m.txt_score >= 0.5)   # fixed threshold 0.5
# m['gap'] = (m.img_score - m.txt_score).abs()
# dev_text = {r['id']: r['text'] for r in C.load_split(DATA_DIR, 'dev')}
# m['text'] = m.id.map(dev_text)
# m[m.disagree].sort_values('gap', ascending=False).head(5)

