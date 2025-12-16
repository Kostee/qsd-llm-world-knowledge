"""
Roberta + Retrieval-Augmented input for Quantified Scope Disambiguation (QSD)
==========================================================================
• 5-fold cross-validation
• Optional retrieval pipeline: fetches top-k relevant passages per sentence
  using Sentence-Transformers + FAISS, prepends to input for RAG
• Default settings tuned for higher-quality retrieval (mpnet embedder, k=3)

Usage:
------
python C_experiments/roberta_rag/roberta_rag_qsd_experiment.py \
    --csv_path C_experiments/data/dataset_for_llms.csv \
    --folds_path C_experiments/data/folds_indices.csv \
    [--corpus_path C_experiments/roberta_rag/rag_corpus/wiki_passages.txt] \
    [--index_path C_experiments/roberta_rag/rag_corpus/wiki_passages.faiss] \
    [--k 3] \
    [--embedder sentence-transformers/multi-qa-mpnet-base-dot-v1] \
    [--device cpu]

Requires: transformers, sentence-transformers, faiss-cpu, scikit-learn, tqdm
"""

import argparse
import os
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer

# Optional RAG dependencies
try:
    import faiss
except ImportError:
    faiss = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ---------- Retrieval helper -------------------------------------------------

def build_or_load_faiss_index(corpus_path: str, index_path: str, model_name: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1", save_index: bool = True):
    assert SentenceTransformer is not None, "sentence-transformers not installed"
    assert faiss is not None, "faiss not installed"

    corpus_path = Path(corpus_path)
    index_path = Path(index_path)
    embedder = SentenceTransformer(model_name)

    if index_path.exists() and index_path.with_suffix('.pkl').exists():
        index = faiss.read_index(str(index_path))
        with open(index_path.with_suffix('.pkl'), 'rb') as f:
            passages = pickle.load(f)
        return embedder, passages, index

    passages = [line.strip() for line in corpus_path.open(encoding='utf-8') if line.strip()]
    embeddings = embedder.encode(passages, batch_size=128, show_progress_bar=True,
                                 convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    if save_index:
        faiss.write_index(index, str(index_path))
        with open(index_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(passages, f)
    return embedder, passages, index


def rag_retrieve(model, index, passages, queries, k=3):
    emb = model.encode(queries, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    _, idxs = index.search(emb, k)
    return [[passages[i] for i in row] for row in idxs]

# ---------- Dataset ----------------------------------------------------------

class QSDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        rag_retriever=None,
        rag_k: int = 3,
    ):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rag_retriever = rag_retriever
        self.rag_k = rag_k

        if self.rag_retriever:
            sentences = dataframe['sentence'].tolist()
            contexts = rag_retriever(sentences)
            # print(f"[RAG DEBUG] sent: {sentences}\n→ context: {contexts}\n")
        else:
            contexts = [[] for _ in range(len(dataframe))]

        for (_, row), ctx in zip(dataframe.iterrows(), contexts):
            text = row['sentence']
            rag_ctx = " ".join(ctx)
            full = f"{rag_ctx} {tokenizer.sep_token} {text}" if rag_ctx else text
            choices = [row['Option A'], row['Option B']]
            label = 0 if row['gold_ans'] == 'A' else 1
            enc = tokenizer(
              [full, full], choices,
              truncation='only_second',
              padding='max_length',
              max_length=self.max_length,
              return_tensors='pt'
            )
            self.examples.append({
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'label': label,
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    labels = torch.tensor([x['label'] for x in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# ---------- Evaluation ------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    total_loss = 0.0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        total_loss += out.loss.item()
        p = torch.argmax(out.logits, dim=-1)
        preds += p.cpu().tolist()
        labs += labels.cpu().tolist()
    loss = total_loss / len(loader)
    acc = accuracy_score(labs, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labs, preds, average='binary')
    return acc, prec, rec, f1, loss

# ---------- Training per fold -----------------------------------------------

def train_fold(model, tokenizer, train_loader, val_loader, device, epochs, lr, fold_dir):
    os.makedirs(fold_dir, exist_ok=True)
    log_file = os.path.join(fold_dir, 'log.txt')
    with open(log_file, 'w') as f_log:
        f_log.write(f"Fold training log in {fold_dir}\n\n")

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        with open(log_file, 'a') as f_log:
            f_log.write(f"Epoch {epoch}/{epochs}\n")

        for batch in tqdm(train_loader, desc=f"Fold {fold_dir} Training E{epoch}", leave=False):
            optimizer.zero_grad()
            out = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            out.loss.backward()
            optimizer.step()
            total_train_loss += out.loss.item()

        avg_loss = total_train_loss / len(train_loader)
        print(f"{fold_dir} Epoch {epoch} train loss: {avg_loss:.4f}")
        with open(log_file, 'a') as f_log:
            f_log.write(f"Train loss: {avg_loss:.4f}\n")

        acc, prec, rec, f1, val_loss = evaluate(model, val_loader, device)
        print(f"{fold_dir} Epoch {epoch} val loss: {val_loss:.4f} acc: {acc:.4f} f1: {f1:.4f}")
        with open(log_file, 'a') as f_log:
            f_log.write(
                f"Val loss: {val_loss:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}\n"
            )

    # Save model
    save_path = os.path.join(fold_dir, 'model')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(log_file, 'a') as f_log:
        f_log.write(f"Model saved to: {save_path}\n")

# ---------- Main -------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="RoBERTa QSD + RAG + 5xCV")
    p.add_argument('--csv_path', required=True)
    p.add_argument('--folds_path', required=True)
    p.add_argument('--corpus_path')
    p.add_argument('--index_path')
    p.add_argument('--embedder', default='sentence-transformers/multi-qa-mpnet-base-dot-v1')
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--model_name', default='roberta-base')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    p.add_argument('--output_dir', default='C_experiments/roberta_rag/')
    args = p.parse_args()

    random.seed(1613); np.random.seed(1613); torch.manual_seed(1613)

    df = pd.read_csv(args.csv_path)

    rag_fn = None
    if args.corpus_path:
        assert faiss and SentenceTransformer, 'Install faiss & sentence-transformers'
        idx = args.index_path or args.corpus_path + '.faiss'
        emb, passg, idxf = build_or_load_faiss_index(args.corpus_path, idx, args.embedder)
        rag_fn = lambda qs: rag_retrieve(emb, idxf, passg, qs, k=args.k)
        print(f"[RAG] Enabled (k={args.k}, embedder={args.embedder})")
    else:
        print("[RAG] Disabled – baseline only.")

    folds_df = pd.read_csv(args.folds_path)
    folds = {c: folds_df[c].dropna().astype(int).tolist() for c in folds_df.columns}

    os.makedirs(args.output_dir, exist_ok=True)
    all_preds, metrics = [], []

    for i, key in enumerate(folds, start=1):
        ti = folds[key]
        tr = [j for k,l in folds.items() if k!=key for j in l]
        train_df = df.loc[tr].reset_index(drop=True)
        test_df  = df.loc[ti].reset_index(drop=False)

        tok = AutoTokenizer.from_pretrained(args.model_name)
        mod = AutoModelForMultipleChoice.from_pretrained(args.model_name).to(args.device)

        tr_ds = QSDDataset(train_df, tok, max_length=args.max_length, rag_retriever=rag_fn, rag_k=args.k)
        va_ds = QSDDataset(test_df,  tok, max_length=args.max_length, rag_retriever=rag_fn, rag_k=args.k)
        tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        va_ld = DataLoader(va_ds, batch_size=args.batch_size, collate_fn=collate_fn)

        fd = os.path.join(args.output_dir, f'fold_{i}')
        train_fold(mod, tok, tr_ld, va_ld, args.device, args.epochs, args.lr, fd)

        acc, prec, rec, f1, _ = evaluate(mod, va_ld, args.device)
        print(f"Fold {i}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
        metrics.append({'fold': i, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1})

        ps = []
        for batch in va_ld:
            out = mod(input_ids=batch['input_ids'].to(args.device), attention_mask=batch['attention_mask'].to(args.device))
            ps.extend(torch.argmax(out.logits, dim=-1).cpu().tolist())
        test_df['prediction'] = ['A' if p==0 else 'B' for p in ps]
        all_preds.append(test_df[['index','sentence','gold_ans','prediction']])

    df_all = pd.concat(all_preds).sort_values('index')
    df_all.to_csv(os.path.join(args.output_dir,'predictions.csv'), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(args.output_dir,'metrics.csv'), index=False)

    print("Done – predictions.csv and metrics.csv written to output_dir.")

if __name__ == '__main__':
    main()