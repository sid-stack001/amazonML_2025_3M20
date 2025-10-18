#!/usr/bin/env python3
"""
price_bert_vit_train_infer.py

- Reuses the BERT regressor pattern and SMAPE loss from the provided notebook.
- Adds a ViT image encoder, fuses text, numeric, and image features.
- Trains on TRAIN_CSV and infers on TEST_CSV to write test.csv with [sample_id, price].

Requirements:
  pip install torch torchvision timm transformers pandas scikit-learn tqdm pillow
"""

import os
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

import timm
from torchvision import transforms
from tqdm import tqdm


# -------------------- USER CONFIG --------------------
IMAGE_PATH_PREFIX = r'F:\amzn_ML_2025\\'  # set to your image folder root
TRAIN_CSV = 'dataset/train_with_local_paths.csv'
TEST_CSV = 'dataset/test_with_local_paths.csv'
OUT_DIR = 'models_out'
TEST_OUT = 'test.csv'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
BATCH_SIZE = 16
EPOCHS = 6
LR = 2e-5
WEIGHT_DECAY = 1e-2
MAX_TEXT_LEN = 128
VAL_SIZE = 0.1
NUM_WORKERS = 4
PIN_MEMORY = True
IMG_SIZE = 224

BERT_MODEL = 'bert-base-uncased'
IMAGE_MODEL = 'vit_base_patch16_224'  # you may switch to 'vit_base_patch16_clip_224.laion2b_ft_in1k'
HIDDEN_DIM = 512
DROPOUT = 0.2

os.makedirs(OUT_DIR, exist_ok=True)
torch.backends.cudnn.benchmark = True


# -------------------- TEXT NUMERIC HELPERS (from notebook pattern) --------------------
VALUE_RE = re.compile(r"Value:\s*([\d\.]+)", re.IGNORECASE)
UNIT_RE = re.compile(r"Unit:\s*([A-Za-z\./ ]+)", re.IGNORECASE)

UNIT_MAP = {
    "fl oz": "fl_oz", "oz": "oz", "ounce": "oz", "ounces": "oz",
    "count": "count", "pkg": "count", "each": "count"
}

def extract_value_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(text, str):
        return None, None
    v_match = VALUE_RE.search(text)
    u_match = UNIT_RE.search(text)
    val = float(v_match.group(1)) if v_match else None
    unit = u_match.group(1).strip().lower() if u_match else None
    return val, unit

def canonical_unit(u: Optional[str]) -> str:
    if not u:
        return "unknown"
    u = u.lower().strip().replace(".", "")
    for k in UNIT_MAP:
        if k in u:
            return UNIT_MAP[k]
    return "other"


# -------------------- DATASET --------------------
class PriceVisionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizerFast, max_len: int,
                 scaler: Optional[StandardScaler], unit_to_idx: dict,
                 img_transform, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.scaler = scaler
        self.unit_to_idx = unit_to_idx
        self.img_transform = img_transform
        self.is_train = is_train

        # Text
        self.texts = self.df['catalog_content'].fillna("").astype(str).tolist()

        # sample_id handling
        if 'sample_id' in self.df.columns:
            self.sids = self.df['sample_id'].astype(np.int64).values
        else:
            self.sids = np.arange(len(self.df), dtype=np.int64)

        # Numeric extraction
        values, units = [], []
        for t in self.texts:
            v, u = extract_value_unit(t)
            values.append(v if v is not None else np.nan)
            units.append(canonical_unit(u))

        self.df['__value__'] = values
        self.df['__unit__'] = units

        # value scaling
        value_array = np.array(self.df['__value__'].fillna(0.0)).reshape(-1, 1)
        self.values_scaled = self.scaler.transform(value_array).astype(np.float32)

        # unit one-hot
        unit_onehot = np.zeros((len(self.df), len(self.unit_to_idx)), dtype=np.float32)
        for i, u in enumerate(self.df['__unit__']):
            idx = self.unit_to_idx.get(u, self.unit_to_idx.get('other', 0))
            unit_onehot[i, idx] = 1.0
        self.unit_onehot = unit_onehot

        # targets (train only)
        if self.is_train:
            if 'price' not in self.df.columns:
                raise ValueError("price column missing for training")
            self.targets = self.df['price'].astype(float).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, row):
        if 'image_local_path' in row and pd.notna(row['image_local_path']) and str(row['image_local_path']).strip():
            p = row['image_local_path']
        elif 'image_link' in row and pd.notna(row['image_link']) and str(row['image_link']).strip():
            p = os.path.basename(str(row['image_link']))
        else:
            return None
        return p if os.path.isabs(p) else os.path.join(IMAGE_PATH_PREFIX, p)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        sid = self.sids[idx]

        # text
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True, padding='max_length',
                             max_length=self.max_len, return_tensors='pt')

        # image
        img_path = self._resolve_image_path(row)
        if img_path is None or not os.path.exists(img_path):
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        else:
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        image_tensor = self.img_transform(img)

        item = {
            'sample_id': torch.tensor(sid, dtype=torch.long),
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'value_scaled': torch.from_numpy(self.values_scaled[idx]),
            'unit_onehot': torch.from_numpy(self.unit_onehot[idx]),
            'image': image_tensor
        }
        if self.is_train:
            item['target'] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


# -------------------- MODEL --------------------
class BertViTRegressor(nn.Module):
    def __init__(self, bert_model_name: str, image_model_name: str,
                 numeric_feat_dim: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        # Text (BERT)
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden = self.bert.config.hidden_size  # typically 768
        self.pre_classifier = nn.Linear(bert_hidden, hidden_dim)
        self.text_drop = nn.Dropout(dropout)

        # Image (timm ViT)
        self.image_encoder = timm.create_model(image_model_name, pretrained=True)
        if hasattr(self.image_encoder, 'reset_classifier'):
            self.image_encoder.reset_classifier(0)
        img_feat_dim = getattr(self.image_encoder, 'num_features', None)
        if img_feat_dim is None:
            img_feat_dim = getattr(getattr(self.image_encoder, 'head', None), 'in_features', 768)

        # Project image to hidden_dim as well
        self.image_proj = nn.Linear(img_feat_dim, hidden_dim)
        self.image_drop = nn.Dropout(dropout)

        # Regressor over [text_hidden, image_hidden, numeric]
        fusion_dim = hidden_dim + hidden_dim + numeric_feat_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Positive output
        self.softplus = nn.Softplus()

    def encode_text(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = out.pooler_output  # (B, H)
        x = self.pre_classifier(pooled)
        x = torch.relu(x)
        x = self.text_drop(x)
        return x

    def encode_image(self, images):
        feat = (self.image_encoder.forward_features(images)
                if hasattr(self.image_encoder, "forward_features") else self.image_encoder(images))
        if isinstance(feat, dict):
            if 'pooled' in feat and feat['pooled'] is not None:
                vec = feat['pooled']
            elif 'x' in feat and feat['x'] is not None:
                x = feat['x']
                vec = x.mean(dim=1) if x.ndim == 3 else x
            else:
                tensors = [v for v in feat.values() if torch.is_tensor(v)]
                x = tensors[0]
                vec = x.mean(dim=1) if x.ndim == 3 else x
        else:
            vec = feat.mean(dim=1) if feat.ndim == 3 else feat
        vec = self.image_proj(vec)
        vec = self.image_drop(vec)
        vec = torch.relu(vec)
        return vec

    def forward(self, input_ids, attention_mask, value_scaled, unit_onehot, images):
        txt = self.encode_text(input_ids, attention_mask)
        img = self.encode_image(images)
        numeric = torch.cat([value_scaled, unit_onehot], dim=1)
        fused = torch.cat([txt, img, numeric], dim=1)
        out = self.head(fused).squeeze(1)
        return self.softplus(out)


# -------------------- LOSS & METRIC (notebook style) --------------------
def smape_loss(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8):
    num = torch.abs(preds - targets)
    denom = (torch.abs(targets) + torch.abs(preds)) / 2.0
    smape = num / (denom + eps)
    return smape.mean()

@torch.no_grad()
def smape_numpy_percent(preds: np.ndarray, targets: np.ndarray, eps: float = 1e-8):
    num = np.abs(preds - targets)
    denom = (np.abs(targets) + np.abs(preds)) / 2.0
    sm = num / (denom + eps)
    return 100.0 * np.mean(sm)


# -------------------- TRAIN / EVAL --------------------
def get_img_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def train_one_epoch(model, loader, optimizer, scheduler, device, scaler_amp):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc='train', leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        value_scaled = batch['value_scaled'].to(device)
        unit_onehot = batch['unit_onehot'].to(device)
        images = batch['image'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler_amp is not None
        ctx = torch.cuda.amp.autocast() if use_amp else torch.autocast('cpu', enabled=False)
        with ctx:
            preds = model(input_ids=input_ids, attention_mask=attention_mask,
                          value_scaled=value_scaled, unit_onehot=unit_onehot, images=images)
            loss = smape_loss(preds, targets)
        if use_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += loss.item() * input_ids.size(0)
        pbar.set_postfix(loss=running / ((pbar.n + 1) * loader.batch_size))
    return running / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds_all, targs_all = [], []
    val_loss = 0.0
    pbar = tqdm(loader, desc='eval', leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        value_scaled = batch['value_scaled'].to(device)
        unit_onehot = batch['unit_onehot'].to(device)
        images = batch['image'].to(device)
        targets = batch['target'].to(device)

        preds = model(input_ids=input_ids, attention_mask=attention_mask,
                      value_scaled=value_scaled, unit_onehot=unit_onehot, images=images)
        loss = smape_loss(preds, targets)
        val_loss += loss.item() * input_ids.size(0)

        preds_all.append(preds.detach().cpu().numpy())
        targs_all.append(targets.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targs_all = np.concatenate(targs_all)
    val_loss = val_loss / len(loader.dataset)
    val_smape_percent = smape_numpy_percent(preds_all, targs_all)
    return val_loss, val_smape_percent, preds_all, targs_all


# -------------------- UTIL --------------------
def fit_numeric_scaler_and_units(df: pd.DataFrame):
    # scaler on value extracted from text
    values_train = []
    units_train = []
    for t in df['catalog_content'].fillna("").astype(str).tolist():
        v, u = extract_value_unit(t)
        values_train.append(v if v is not None else 0.0)
        units_train.append(canonical_unit(u))
    scaler = StandardScaler()
    scaler.fit(np.array(values_train, dtype=np.float32).reshape(-1, 1))

    # unit set
    unique_units = ['fl_oz', 'oz', 'count', 'other', 'unknown']
    unit_to_idx = {u: i for i, u in enumerate(unique_units)}
    return scaler, unit_to_idx


# -------------------- MAIN --------------------
def main():
    global IMAGE_PATH_PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default=TRAIN_CSV)
    parser.add_argument('--test_csv', type=str, default=TEST_CSV)
    parser.add_argument('--image_prefix', type=str, default=IMAGE_PATH_PREFIX)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--max_len', type=int, default=MAX_TEXT_LEN)
    parser.add_argument('--bert_model', type=str, default=BERT_MODEL)
    parser.add_argument('--image_model', type=str, default=IMAGE_MODEL)
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    parser.add_argument('--test_out', type=str, default=TEST_OUT)
    args = parser.parse_args()


    IMAGE_PATH_PREFIX = args.image_prefix

    device = torch.device(DEVICE)
    print('Device:', device)

    # Load CSV
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Missing {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    train_df = train_df.dropna(subset=['price']).reset_index(drop=True)
    train_df['catalog_content'] = train_df['catalog_content'].fillna("").astype(str)

    # Train/Val split
    tr_df, va_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=SEED, shuffle=True)

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model)

    # Numeric scaler and unit map from train split
    scaler, unit_to_idx = fit_numeric_scaler_and_units(tr_df)

    # Transforms
    img_tf = get_img_transform()

    # Datasets / Loaders
    numeric_feat_dim = 1 + len(unit_to_idx)  # scaled value + unit onehot
    tr_ds = PriceVisionDataset(tr_df, tokenizer, max_len=args.max_len,
                               scaler=scaler, unit_to_idx=unit_to_idx,
                               img_transform=img_tf, is_train=True)
    va_ds = PriceVisionDataset(va_df, tokenizer, max_len=args.max_len,
                               scaler=scaler, unit_to_idx=unit_to_idx,
                               img_transform=img_tf, is_train=True)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Model
    model = BertViTRegressor(args.bert_model, args.image_model,
                             numeric_feat_dim=numeric_feat_dim,
                             hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    model.to(device)

    # Optimizer & Scheduler (notebook-style)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(tr_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.05 * total_steps)),
        num_training_steps=total_steps
    )

    # AMP scaler
    use_amp = torch.cuda.is_available()
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None

    # Train
    best_smape = float('inf')
    best_path = Path(args.out_dir) / "best_model.pt"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(model, tr_loader, optimizer, scheduler, device, scaler_amp)
        val_loss, val_smape_percent, _, _ = eval_epoch(model, va_loader, device)
        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | Val SMAPE(%): {val_smape_percent:.4f}")

        if val_smape_percent < best_smape:
            best_smape = val_smape_percent
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'unit_to_idx': unit_to_idx,
                'tokenizer_name': args.bert_model,
                'image_model_name': args.image_model
            }, best_path)
            print(f"Saved best to {best_path} (Val SMAPE {val_smape_percent:.4f}%)")

    print("Best Val SMAPE(%):", best_smape)

    # -------------------- INFERENCE ON TEST --------------------
    if not os.path.exists(args.test_csv):
        print(f"Test CSV not found at {args.test_csv}; skipping inference.")
        return

    # Load best
    if not best_path.exists():
        raise FileNotFoundError("No trained checkpoint found for inference")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Prepare test df
    test_df = pd.read_csv(args.test_csv)
    if 'catalog_content' not in test_df.columns:
        test_df['catalog_content'] = ""
    test_df['catalog_content'] = test_df['catalog_content'].fillna("").astype(str)
    # enforce sample_id
    if 'sample_id' not in test_df.columns:
        test_df['sample_id'] = np.arange(len(test_df), dtype=np.int64)

    # Reuse tokenizer, scaler, unit map from training
    scaler = ckpt['scaler']
    unit_to_idx = ckpt['unit_to_idx']
    tokenizer = BertTokenizerFast.from_pretrained(ckpt['tokenizer_name'])

    test_ds = PriceVisionDataset(test_df, tokenizer, max_len=args.max_len,
                                 scaler=scaler, unit_to_idx=unit_to_idx,
                                 img_transform=img_tf, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    preds = []
    sids = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='inference', leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            value_scaled = batch['value_scaled'].to(device)
            unit_onehot = batch['unit_onehot'].to(device)
            images = batch['image'].to(device)
            sid = batch['sample_id'].cpu().numpy().astype(np.int64)

            pr = model(input_ids=input_ids, attention_mask=attention_mask,
                       value_scaled=value_scaled, unit_onehot=unit_onehot, images=images)
            pr = pr.detach().cpu().numpy().astype(np.float64)  # ensure float
            # enforce positive
            pr = np.clip(pr, a_min=0.01, a_max=None)

            preds.append(pr)
            sids.append(sid)

    preds = np.concatenate(preds)
    sids = np.concatenate(sids)

    sub = pd.DataFrame({'sample_id': sids.astype(np.int64), 'price': preds.astype(np.float64)})
    # if original file has ordering, respect it
    if 'sample_id' in test_df.columns:
        sub = sub.set_index('sample_id').reindex(test_df['sample_id'].astype(np.int64)).reset_index()
    sub.to_csv(args.test_out, index=False)
    print(f"Wrote {args.test_out}")


if __name__ == "__main__":
    main()
