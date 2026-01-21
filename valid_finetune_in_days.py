import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import stft
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iq_to_spectrogram(iq_data, nperseg=256, noverlap=128):
    """Convert IQ (complex or stacked real/imag) to spectrogram using STFT.

    Matches the training pipeline in model_ZNAX/main copy_in_days.py.
    """
    iq_data = np.asarray(iq_data)
    if np.iscomplexobj(iq_data):
        complex_signal = iq_data
    else:
        # Support stacked format: [I... I, Q... Q]
        if iq_data.ndim != 1:
            iq_data = iq_data.reshape(-1)
        half = iq_data.shape[0] // 2
        I = iq_data[:half]
        Q = iq_data[half:half + half]
        complex_signal = I + 1j * Q
    f, t, Zxx = stft(complex_signal, nperseg=nperseg, noverlap=noverlap)
    spectrogram = np.abs(Zxx)
    return spectrogram


def get_fold_data(spectrograms, labels, fold_idx=0, k=5, seed=42):
    # ensure numpy
    if hasattr(spectrograms, 'cpu'):
        spectrograms = spectrograms.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    if len(labels.shape) > 1:
        labels = labels.flatten()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(skf.split(spectrograms, labels))
    train_idx, test_idx = folds[fold_idx]
    train_data = spectrograms[train_idx]
    test_data = spectrograms[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    return train_data, test_data, train_labels, test_labels


class SupervisedH5Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data.astype(np.float32)
        self.labels = labels.flatten()
        self.transform = transform
        # truncate if mismatch
        if len(self.data) != len(self.labels):
            min_len = min(len(self.data), len(self.labels))
            self.data = self.data[:min_len]
            self.labels = self.labels[:min_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


# Minimal ViT components (match definitions used in training)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # compute number of patches
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)])

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        for layer in self.encoder:
            x = layer(x)
        return x[:, 0]


class VisionTransformerWithClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.vit = pretrained_model
        self.classifier = nn.Linear(pretrained_model.cls_token.shape[-1], num_classes)

    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)


def load_checkpoint_to_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device).float()
        y = y.to(device).long()
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def main(fold_idx=0, days='3,4', ckpt=None, batch_size=64, device_num=0, gpus='0', seed=42):
    set_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')

    if isinstance(days, str) and ',' in days:
        days_list = [int(d) for d in days.split(',')]
    else:
        days_list = [int(days)] if isinstance(days, str) else days

    # load data same as training pipeline (complex IQ, day list supported)
    f = np.load(f'dataset/self_lora/day9.npz', allow_pickle=True)['sample'][:]
    labels = np.load(f'dataset/self_lora/day9.npz', allow_pickle=True)['labels'][:]


    data = f
    spectrograms = np.array([iq_to_spectrogram(x, nperseg=256, noverlap=128) for x in data])
    spectrograms = spectrograms.reshape(-1, 256, 65, 1)

    # labels = np.asarray(label_t).flatten()
    try:
        labels = labels.astype(np.int64)
    except Exception:
        labels = np.array(labels, dtype=np.int64)
    if labels.min() == 1:
        labels = labels - 1
    elif labels.min() < 0:
        unique = np.unique(labels)
        mapping = {v: i for i, v in enumerate(unique)}
        labels = np.array([mapping[v] for v in labels], dtype=np.int64)

    # compute mean/std as in training
    mean = np.mean(spectrograms)
    std = np.std(spectrograms)

    # get fold split
    _, test_data, _, test_labels = get_fold_data(spectrograms, labels, fold_idx=fold_idx, k=5, seed=seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    test_dataset = SupervisedH5Dataset(test_data, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # build model matching fine-tune
    spectrogram_size = (256, 65)
    patch_size = (16, 8)
    vit = VisionTransformer(spectrogram_size, patch_size, 1, 128, 6, 8, 256)

    # Determine checkpoint path (prefer best then final) -- match training naming
    save_dir = f'./model_ZNAX/save{fold_idx}'
    if ckpt is None:
        ckpt_path = os.path.join(save_dir, f'best_finetune_fold{fold_idx}_{", ".join(map(str,days_list))}.pth') if len(days_list)<=1 else os.path.join(save_dir, f'best_finetune_fold{fold_idx}_[{", ".join(map(str,days_list))}].pth')
        
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(save_dir, f'final_finetune_fold{fold_idx}_{", ".join(map(str,days_list))}.pth') if len(days_list)<=1 else os.path.join(save_dir, f'final_finetune_fold{fold_idx}_[{", ".join(map(str,days_list))}].pth')
        
    else:
        ckpt_path = ckpt

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    # Heuristic: inspect checkpoint to infer classifier output size so we instantiate
    # the evaluation model with the correct num_classes. This prevents the case
    # where the classifier in the script is created with a different output size
    # and thus its weights are skipped when loading (leaving random weights).
    try:
        ckpt_obj = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt_obj, dict):
            # common keys: 'model_state_dict' or 'state_dict' or raw state_dict
            state = ckpt_obj.get('model_state_dict', ckpt_obj.get('state_dict', ckpt_obj))
        else:
            state = None
    except Exception:
        state = None

    num_classes_from_ckpt = None
    if isinstance(state, dict):
        # look for classifier weight key
        for k, v in state.items():
            if k.endswith('classifier.weight') or k.endswith('classifier.weight_orig') or '.classifier.weight' in k:
                try:
                    num_classes_from_ckpt = int(v.shape[0])
                    break
                except Exception:
                    pass

    if num_classes_from_ckpt is not None:
        num_classes = num_classes_from_ckpt
    else:
        num_classes = int(len(np.unique(test_labels)))

    model = VisionTransformerWithClassifier(vit, num_classes).to(device)
    load_checkpoint_to_model(model, ckpt_path, device)

    acc = evaluate(model, test_loader, device)
    print(f"Fold={fold_idx}, days={days_list}, ckpt={ckpt_path}, test_samples={len(test_dataset)}, ACC={acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_idx', type=int, default=0)
    parser.add_argument('--days', type=str, default='4')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(fold_idx=args.fold_idx, days=args.days, ckpt=args.ckpt, batch_size=args.batch_size, device_num=args.device_num, gpus=args.gpus, seed=args.seed)
