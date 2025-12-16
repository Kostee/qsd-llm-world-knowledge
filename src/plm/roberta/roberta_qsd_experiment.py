# run by:
# cd C:\Users\Jakub\Documents\GitHub\scope-ambliguity
# python C_experiments/roberta/roberta_qsd_experiment.py --csv_path C_experiments/data/dataset_for_llms.csv

import argparse
import os
import random

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from torch.optim import AdamW

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


class QSDDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for _, row in dataframe.iterrows():
            sent = row['sentence']
            choices = [row['Option A'], row['Option B']]
            label = 0 if row['gold_ans'] == 'A' else 1
            enc = tokenizer(
                [sent, sent],
                [choices[0], choices[1]],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.examples.append({
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'label': label
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    return acc, precision, recall, f1, avg_loss, all_preds, all_labels


def train_fold(model, tokenizer, train_loader, val_loader, device, 
               epochs=10, lr=2e-5, fold_dir=''):
    os.makedirs(fold_dir, exist_ok=True)
    log_file = os.path.join(fold_dir, 'log.txt')
    with open(log_file, 'w') as f_log:
        f_log.write(f"Fold training log in {fold_dir}\n\n")

    optimizer = AdamW(model.parameters(), lr=lr)
    fold_metrics = []

    for epoch in range(1, epochs + 1):
        # Logging epoch start
        print(f"Fold {fold_dir}: Epoch {epoch}/{epochs}")
        with open(log_file, 'a') as f_log:
            f_log.write(f"Epoch {epoch}/{epochs}\n")

        # Training step
        model.train()
        total_train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"  Train loss: {avg_train_loss:.4f}")
        with open(log_file, 'a') as f_log:
            f_log.write(f"Train loss: {avg_train_loss:.4f}\n")

        # Validation step
        acc, precision, recall, f1, val_loss, _, _ = evaluate(model, val_loader, device)
        print(f"  Val loss: {val_loss:.4f}, Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
        with open(log_file, 'a') as f_log:
            f_log.write(
                f"Val loss: {val_loss:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, F1: {f1:.4f}\n"
            )

        fold_metrics.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # Save fold model
    model_save_path = os.path.join(fold_dir, 'model')
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model for {fold_dir} saved to: {model_save_path}")
    with open(log_file, 'a') as f_log:
        f_log.write(f"Model saved to: {model_save_path}\n")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Roberta QSD 5-Fold Cross-Validation'
    )
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--folds_path', type=str,
                        default='C_experiments/data/folds_indices.csv')
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--device', type=str,
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--output_dir', type=str,
                        default='C_experiments/roberta/')
    args = parser.parse_args()

    # Seeds for reproducibility
    random.seed(1613)
    torch.manual_seed(1613)
    np.random.seed(1613)

    # Load dataset and folds
    df = pd.read_csv(args.csv_path)
    folds_df = pd.read_csv(args.folds_path)

    # Prepare folds indices
    folds = {}
    for col in folds_df.columns:
        folds[col] = folds_df[col].dropna().astype(int).tolist()

    os.makedirs(args.output_dir, exist_ok=True)
    all_preds = []
    metrics_all_folds = []

    # Cross-validation loop
    for i in range(1, len(folds) + 1):
        test_key = f'Fold_{i}'
        test_idxs = folds[test_key]
        train_idxs = [idx for key, lst in folds.items() if key != test_key for idx in lst]

        train_df = df.loc[train_idxs].reset_index(drop=True)
        test_df = df.loc[test_idxs].reset_index(drop=False)

        # Tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name
        ).to(args.device)

        # Datasets and loaders
        train_dataset = QSDDataset(train_df, tokenizer, max_length=args.max_length)
        val_dataset = QSDDataset(test_df, tokenizer, max_length=args.max_length)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            collate_fn=collate_fn, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size,
            collate_fn=collate_fn
        )

        # Directory for this fold
        fold_dir = os.path.join(args.output_dir, f'fold_{i}')

        # Train and collect metrics
        fold_metrics = train_fold(
            model, tokenizer, train_loader, val_loader,
            device=args.device, epochs=args.epochs,
            lr=args.lr, fold_dir=fold_dir
        )

        # Final evaluation and predictions
        acc, precision, recall, f1, val_loss, preds, labels = (
            evaluate(model, val_loader, args.device)
        )

        # Save fold metrics
        for m in fold_metrics:
            m['fold'] = i
        metrics_all_folds.extend(fold_metrics)

        # Collect predictions with original indices and gold labels
        preds_df = test_df.copy()
        preds_df['prediction'] = ['A' if p == 0 else 'B' for p in preds]
        all_preds.append(preds_df[['index', 'sentence', 'gold_ans', 'prediction']])

    # Save all predictions
    all_preds_df = pd.concat(all_preds).sort_values('index')
    all_preds_df.to_csv(
        os.path.join(args.output_dir, 'predictions.csv'), index=False
    )

    # Save metrics for all folds
    metrics_df = pd.DataFrame(metrics_all_folds)
    metrics_df.to_csv(
        os.path.join(args.output_dir, 'metrics.csv'), index=False
    )

    print("Cross-validation complete.")


if __name__ == '__main__':
    main()
