#Necessary Imports

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
import os
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import random
from itertools import combinations
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import argparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def get_fold_data(spectrograms, labels, fold_idx=0, k=5, seed=42):
    """
    5折交叉验证数据分割函数
    Args:
        spectrograms: 频谱图数据
        labels: 标签数据
        fold_idx: 当前折的索引 (0-4)
        k: 折数，默认5
        seed: 随机种子
    Returns:
        train_data, test_data, train_labels, test_labels
    """
    # 确保输入是numpy array
    if hasattr(spectrograms, 'cpu'):
        spectrograms = spectrograms.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    
    # 处理labels的维度问题
    if len(labels.shape) > 1:
        labels = labels.flatten()
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    # 生成所有折的索引
    folds = list(skf.split(spectrograms, labels))
    
    # 获取当前折的训练和测试索引
    train_idx, test_idx = folds[fold_idx]
    
    # 分割数据
    train_data = spectrograms[train_idx]
    test_data = spectrograms[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    
    return train_data, test_data, train_labels, test_labels

# Mount Google Drive / Local Drive
# drive.mount('/content/drive') #Replace with your actual path

# Patch Embedding
def main(fold_idx=0, seed=42, days='3,4'):
    set_seed(seed)
    print(f"Running fold {fold_idx + 1}/5 with seed {seed}")
    
    # Initialize TensorBoard writer
    log_dir = f'./model_ZNAX/runs/fold_{fold_idx}_seed_{seed}_days_{days}'
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Custom Dataset Class for Contrastive Learning (returns triplets)
    class ContrastiveH5Dataset(Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = data.astype(np.float32)
            self.labels = labels.flatten()  # Flatten the labels to make it a 1D array
            self.transform = transform
            
            # 确保数据和标签长度匹配
            if len(self.data) != len(self.labels):
                print(f"WARNING: Contrastive data and labels length mismatch!")
                print(f"Data length: {len(self.data)}, Labels length: {len(self.labels)}")
                # 取较小的长度以避免索引越界
                min_len = min(len(self.data), len(self.labels))
                self.data = self.data[:min_len]
                self.labels = self.labels[:min_len]
                print(f"Truncated to length: {min_len}")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            if idx >= len(self.data) or idx >= len(self.labels):
                raise IndexError(f"Index {idx} out of bounds. Data length: {len(self.data)}, Labels length: {len(self.labels)}")
            
            # Select the sample and its label
            sample = self.data[idx]
            label = self.labels[idx]

            # Apply transformations
            if self.transform:
                sample = self.transform(sample)

                # Create a positive pair (same label)
                # Simply use the same sample for now
                sample2 = self.data[idx]  # Positive pair (same class)
                sample2 = self.transform(sample2)

                # Randomly select a negative pair (different label)
                # Find an index with a different label
                neg_indices = np.where(self.labels != label)[0]  # 找到所有不同标签的索引
                if len(neg_indices) > 0:
                    neg_idx = np.random.choice(neg_indices)  # 随机选择一个负样本
                    sample_neg = self.data[neg_idx]
                    sample_neg = self.transform(sample_neg)
                    pair_label = torch.tensor(0.0)  # 确保是负样本
                else:
                    # 如果没有负样本，使用当前样本作为正样本
                    sample_neg = sample2
                    pair_label = torch.tensor(1.0)

                return sample, sample2, pair_label  # Positive pair (same class)

            return sample, label
    
    # Custom Dataset Class for Supervised Learning (returns pairs)
    class SupervisedH5Dataset(Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = data.astype(np.float32)
            self.labels = labels.flatten()  # Flatten labels for consistency
            self.transform = transform
            
            # 确保数据和标签长度匹配
            if len(self.data) != len(self.labels):
                print(f"WARNING: Data and labels length mismatch!")
                print(f"Data length: {len(self.data)}, Labels length: {len(self.labels)}")
                # 取较小的长度以避免索引越界
                min_len = min(len(self.data), len(self.labels))
                self.data = self.data[:min_len]
                self.labels = self.labels[:min_len]
                print(f"Truncated to length: {min_len}")

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            if idx >= len(self.data) or idx >= len(self.labels):
                raise IndexError(f"Index {idx} out of bounds. Data length: {len(self.data)}, Labels length: {len(self.labels)}")
            
            sample = self.data[idx]
            label = self.labels[idx]

            if self.transform:
                sample = self.transform(sample)

            return sample, label
    
    # ================================
    # 统一的数据加载和预处理
    # ================================
    def iq_to_spectrogram(iq_data, nperseg=256, noverlap=128):
        """Convert IQ (complex or stacked real/imag) to spectrogram using STFT."""
        iq_data = np.asarray(iq_data)
        if np.iscomplexobj(iq_data):
            complex_signal = iq_data
        else:
            iq_data = iq_data.reshape(-1)
            if iq_data.size < 2:
                raise ValueError(f"iq_data too short: {iq_data.size}")
            half = iq_data.size // 2
            iq_data = iq_data[: 2 * half]
            I = iq_data[:half]
            Q = iq_data[half:]
            complex_signal = I + 1j * Q

        f, t, Zxx = stft(complex_signal, nperseg=nperseg, noverlap=noverlap)
        spectrogram = np.abs(Zxx)
        return spectrogram
    
    # 加载数据集（只加载一次）
    # dataset_path = './dataset/dataset_training_no_aug.h5'
    # with h5py.File(dataset_path, 'r') as f:
    #     data = np.array(f['data'])    
    #     labels = np.array(f['label']) 

    if len(days) == 1:
        days = int(days)
    else:
        days = [int(d) for d in days.split(',')]
        
    f = np.concatenate([np.load('dataset/self_lora/day{}.npz'.format(i))['sample'][:] for i in days], axis=0) if isinstance(days, list) else np.load('dataset/self_lora/day{}.npz'.format(days), allow_pickle=True)['sample'][:]
    
    label_t = np.concatenate([np.load('dataset/self_lora/day{}.npz'.format(i))['labels'][:] for i in days], axis=0) if isinstance(days, list) else np.load('dataset/self_lora/day{}.npz'.format(days), allow_pickle=True)['labels'][:]

    # IMPORTANT: keep IQ as complex. The previous real/imag concatenation breaks IQ parsing
    # and produces incorrect spectrograms.
    data = f
    labels = label_t
    # 添加调试信息
    print(f"Original data shape: {data.shape}")
    print(f"Original labels shape: {np.asarray(labels).shape}")

    # 转换为频谱图
    spectrograms = []
    for iq_sample in data:
        spectrogram = iq_to_spectrogram(iq_sample, nperseg=256, noverlap=128)
        spectrograms.append(spectrogram)

    spectrograms = np.array(spectrograms)
    print(f"Shape of spectrograms: {spectrograms.shape}")

    # 重新整形数据
    spectrograms = spectrograms.reshape(-1, 256, 65, 1)
    labels = np.asarray(labels).flatten()
    # Ensure labels are 0-based integers. If labels start from 1, convert to 0-based.
    try:
        labels = labels.astype(np.int64)
    except Exception:
        labels = np.array(labels, dtype=np.int64)
    if labels.min() == 1:
        labels = labels - 1
    elif labels.min() < 0:
        # If negative labels are already present, remap unique labels to 0..C-1
        unique = np.unique(labels)
        mapping = {v: i for i, v in enumerate(unique)}
        labels = np.array([mapping[v] for v in labels], dtype=np.int64)
    
    # 添加调试信息
    print(f"After processing - spectrograms shape: {spectrograms.shape}")
    print(f"After processing - labels shape: {labels.shape}")
    print(f"Labels range: min={labels.min()}, max={labels.max()}")
    
    # 检查数据和标签数量是否匹配
    if len(spectrograms) != len(labels):
        print(f"ERROR: Data and labels count mismatch!")
        print(f"Spectrograms: {len(spectrograms)}, Labels: {len(labels)}")
        # 尝试修复：如果labels是类别标签而不是每个样本的标签
        if len(labels) < len(spectrograms):
            print("Attempting to fix: creating labels for each sample...")
            # 假设每个类别有相同数量的样本
            samples_per_class = len(spectrograms) // len(labels)
            new_labels = []
            for class_idx, class_label in enumerate(labels):
                new_labels.extend([class_label] * samples_per_class)
            # 处理剩余样本
            remaining_samples = len(spectrograms) - len(new_labels)
            if remaining_samples > 0:
                # 将剩余样本分配给最后一个类别
                new_labels.extend([labels[-1]] * remaining_samples)
            labels = np.array(new_labels)
            print(f"Fixed labels shape: {labels.shape}")
    
    print(f"Final check - spectrograms: {spectrograms.shape}, labels: {labels.shape}")

    # 计算数据集的均值和标准差
    mean = np.mean(spectrograms)
    std = np.std(spectrograms)

    # 5折交叉验证数据划分（统一划分，两个阶段都使用）
    train_data, test_data, train_labels, test_labels = get_fold_data(
        spectrograms, labels, fold_idx=fold_idx, k=5, seed=seed)
    
    indices = np.random.RandomState(seed).permutation(len(train_data))
        
    # 三分割：60% 对比训练, 20% 对比验证, 20% 监督训练
    contrastive_train_size = int(0.6 * len(train_data))
    contrastive_val_size = int(0.2 * len(train_data))
    supervised_train_size = len(train_data) - contrastive_train_size - contrastive_val_size
    
    contrastive_train_idx = indices[:contrastive_train_size]
    contrastive_val_idx = indices[contrastive_train_size:contrastive_train_size + contrastive_val_size]
    supervised_train_idx = indices[contrastive_train_size + contrastive_val_size:]
    
    # 对比学习数据
    pretrain_train_data = train_data[contrastive_train_idx]
    pretrain_val_data = train_data[contrastive_val_idx]
    pretrain_train_labels = train_labels[contrastive_train_idx]
    pretrain_val_labels = train_labels[contrastive_val_idx]
    
    # 监督学习数据（独立于对比学习数据）
    supervised_train_data = train_data[supervised_train_idx]
    supervised_train_labels = train_labels[supervised_train_idx]
    
    
    print(f"Training data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    print(f"Training labels shape: {train_labels.shape}, Test labels shape: {test_labels.shape}")
    class PatchEmbedding(nn.Module):
        def __init__(self, img_size, patch_size, in_channels, embed_dim):
            super().__init__()
            self.patch_size = patch_size
            self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)  
            return x
        
    
    class PatchEmbedding(nn.Module):
        def __init__(self, img_size, patch_size, in_channels, embed_dim):
            super().__init__()
            self.patch_size = patch_size
            self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        def forward(self, x):
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)  
            return x



    # Multi-head Self-Attention

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


    # Transformer Encoder Block

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
                nn.Linear(mlp_dim, embed_dim)
            )

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


    # Vision Transformer
    class VisionTransformer(nn.Module):
        def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim):
            super().__init__()
            self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
            self.encoder = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
            self.learnable_margins = LearnableMargins()

        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
            for layer in self.encoder:
                x = layer(x)
            return x[:, 0]  # Use CLS token embedding for contrastive loss

        def get_margins(self):
            return self.learnable_margins()
    
    if not os.path.exists('./model_ZNAX/save'+str(fold_idx)+'/vit_model_finetuned.pth'):
        # ================================
        # 第一阶段：预训练（Contrastive Learning）
        # ================================
        print("\n" + "="*50)
        print("Phase 1: Contrastive Pretraining")
        print("="*50)
        


        # Dataset Preparation with Data Augmentation (Optional and applicable only for training data)
        train_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            #transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
            #transforms.RandomCrop(32, padding=4),  # Random crop with padding
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Validation 
        val_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load CIFAR-10 training data
        full_train_dataset = datasets.CIFAR10(root='./model_ZNAX/save/data', train=True, download=True, transform=train_transform)

        # Split training dataset into train and validation subsets
        train_size = int(0.8 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        # Pair balancing for validation dataset (no augmentation applied)
        def create_balanced_pairs(dataset):
            """
            Create balanced pairs of positive and negative samples from a dataset.
            Args:
                dataset: PyTorch dataset object (e.g., validation dataset).
            Returns:
                list of (img1, img2, label) pairs
            """
            class_to_images = {}
            for idx, (image, label) in enumerate(dataset):
                if label not in class_to_images:
                    class_to_images[label] = []
                class_to_images[label].append((image, idx))

            positive_pairs = []
            negative_pairs = []

            #Pseudo labelling
            # Create positive pairs (within the same class)
            for class_label, images in class_to_images.items():
                if len(images) > 1:
                    for (img1, _), (img2, _) in combinations(images, 2):
                        positive_pairs.append((img1, img2, 1))

            # Create negative pairs (from different classes)
            all_classes = list(class_to_images.keys())
            for class1, class2 in combinations(all_classes, 2):
                for (img1, _), (img2, _) in zip(class_to_images[class1], class_to_images[class2]):
                    negative_pairs.append((img1, img2, 0))

            # Balance the number of pairs
            num_pairs = min(len(positive_pairs), len(negative_pairs))
            positive_pairs = random.sample(positive_pairs, num_pairs)
            negative_pairs = random.sample(negative_pairs, num_pairs)

            all_pairs = positive_pairs + negative_pairs
            random.shuffle(all_pairs)

            return all_pairs

        # Create balanced pairs for validation
        validation_pairs = create_balanced_pairs(val_dataset)

        # Wrap the pairs into a DataLoader
        class PairDataset(torch.utils.data.Dataset):
            def __init__(self, pairs):
                self.pairs = pairs

            def __len__(self):
                return len(self.pairs)

            def __getitem__(self, idx):
                img1, img2, label = self.pairs[idx]
                return img1, img2, label


        # Create validation and test DataLoader
        val_loader = DataLoader(PairDataset(validation_pairs), batch_size=64, shuffle=False)

        # Train and test loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_dataset = datasets.CIFAR10(root='./model_ZNAX/save/data', train=False, download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Dual Margin Contrastive Loss Function

        class LearnableMargins(nn.Module):
            def __init__(self, initial_m1=0.6, initial_m2=1.6): # m1=0.0, and m2=0.6 (As training proceeds, m1 will increase and m2 will decrease)
                super().__init__()
                self.m1 = nn.Parameter(torch.tensor(initial_m1, dtype=torch.float32))
                self.m2 = nn.Parameter(torch.tensor(initial_m2, dtype=torch.float32))




            def forward(self):
                # Ensure m1 < m2 by enforcing constraints
                return torch.clamp(self.m1, min=0), torch.clamp(self.m2, min=self.m1 + 0.1)


        def dual_margin_contrastive_loss(emb1, emb2, label, m1, m2):
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            D = F.pairwise_distance(emb1, emb2)
            loss_pos = label * torch.pow(torch.clamp(D - m1, min=0), 2)
            loss_neg = (1 - label) * torch.pow(torch.clamp(m2 - D, min=0), 2)
            return 30.0 * torch.mean(loss_pos + loss_neg)

        # Initialize device, model, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VisionTransformer(32, 8, 3, 128, 6, 8, 256).to(device)
        #(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim):

        # Exclude learnable_margins parameters from the main parameter group
        base_params = [
            p for name, p in model.named_parameters() if "learnable_margins" not in name
        ]

        optimizer = AdamW(
            [{'params': base_params},  # Regular model parameters
            {'params': model.learnable_margins.parameters(), 'lr': 1e-3}],  # Higher learning rate for margins
            lr=4e-4, weight_decay=1e-1
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        # Store embeddings during the training loop
        cifar_embeddings = []

        # Training Loop
        for epoch in range(40):
            model.train()
            total_train_loss = 0
            num_batches = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                batch_size = images.size(0)

                # Create pairs within the batch
                idx = torch.randperm(batch_size)
                img1, img2 = images, images[idx]
                pair_labels = (labels == labels[idx]).float().to(device)

                # Forward pass
                emb1 = model(img1)
                emb2 = model(img2)

                # Save embeddings for CIFAR10
                cifar_embeddings.append(emb1.cpu().detach().numpy())  
                cifar_embeddings.append(emb2.cpu().detach().numpy())  

                # Get learnable margins
                m1, m2 = model.get_margins()

                # Compute contrastive loss
                loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1
                
                # Record batch-level metrics to TensorBoard
                global_step = epoch * len(train_loader) + num_batches
                writer.add_scalar('CIFAR_Pretraining/Batch_Loss', loss.item(), global_step)
                writer.add_scalar('CIFAR_Pretraining/Margin_m1', m1.item(), global_step)
                writer.add_scalar('CIFAR_Pretraining/Margin_m2', m2.item(), global_step)

            scheduler.step()

            # Validation
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_img1, val_img2, val_labels in val_loader:
                    val_img1, val_img2, val_labels = (
                        val_img1.to(device),
                        val_img2.to(device),
                        val_labels.to(device),
                    )

                    # Forward pass for validation
                    val_emb1 = model(val_img1)
                    val_emb2 = model(val_img2)

                    # Get learnable margins
                    val_m1, val_m2 = model.get_margins()

                    # Compute validation contrastive loss
                    val_loss = dual_margin_contrastive_loss(
                        val_emb1, val_emb2, val_labels, val_m1, val_m2
                    )
                    total_val_loss += val_loss.item()

            # Logging losses
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Record epoch-level metrics to TensorBoard
            writer.add_scalar('CIFAR_Pretraining/Epoch_Train_Loss', avg_train_loss, epoch)
            writer.add_scalar('CIFAR_Pretraining/Epoch_Val_Loss', avg_val_loss, epoch)
            writer.add_scalar('CIFAR_Pretraining/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            print(
                f"Epoch [{epoch+1}/30], Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
            )

        # After training, save embeddings and model


        cifar_embeddings = np.array(cifar_embeddings)  # Convert the list of embeddings to a numpy array

        embedding_save_path = './model_ZNAX/save'+str(fold_idx)+'/cifar_embeddings.npy'  # Path to save embeddings --- Replace with your actual path
        if not os.path.exists('./model_ZNAX/save'+str(fold_idx)):
            os.makedirs('./model_ZNAX/save'+str(fold_idx))
        np.save(embedding_save_path, cifar_embeddings)
        print(f"Embeddings saved to {embedding_save_path}")

        # Specify the save path for the model
        save_path = './model_ZNAX/save'+str(fold_idx)+'/vit_model.pth' #Define path to save---Replace with your actual path

        # Save the model's state_dictq
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


        #Evaluation on Test Data
        model.eval()  
        total_test_loss = 0

        with torch.no_grad():
            for test_images, batch_test_labels in test_loader:
                test_images, batch_test_labels = test_images.to(device), batch_test_labels.to(device)
                batch_size = test_images.size(0)

                # Create pairs within the test batch
                idx = torch.randperm(batch_size)
                test_img1, test_img2 = test_images, test_images[idx]
                test_pair_labels = (batch_test_labels == batch_test_labels[idx]).float().to(device)

                # Forward pass for test data
                test_emb1 = model(test_img1)
                test_emb2 = model(test_img2)

                # Get learnable margins
                test_m1, test_m2 = model.get_margins()

                # Compute test contrastive loss
                test_loss = dual_margin_contrastive_loss(test_emb1, test_emb2, test_pair_labels, test_m1, test_m2)
                total_test_loss += test_loss.item()

        # Average test loss
        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}, Final m1: {test_m1.item():.4f}, Final m2: {test_m2.item():.4f}")


        #Load the Pretrained model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VisionTransformer(32, 8, 3, 128, 6, 8, 256).to(device)  

        save_path = './model_ZNAX/save'+str(fold_idx)+'/vit_model.pth' # Replace with your actual path
        model.load_state_dict(torch.load(save_path))
        model.eval() 

        # #If need to evaluate-without threshold, with exact margins

        # total_pairs = 0
        # correct_pairs = 0

        # with torch.no_grad():
        #     for images, labels in test_loader:
        #         images, labels = images.to(device), labels.to(device)
        #         batch_size = images.size(0)

        #         # Create pairs within the batch
        #         idx = torch.randperm(batch_size)
        #         img1, img2 = images, images[idx]
        #         pair_labels = (labels == labels[idx]).float().to(device)  # 1 if same class, 0 otherwise

        #         # Get embeddings
        #         emb1 = model(img1)
        #         emb2 = model(img2)

        #         # Compute pairwise distance
        #         distances = F.pairwise_distance(emb1, emb2)

        #         # Determine correct predictions based on margins
        #         predictions = torch.where(pair_labels == 1, distances < m1, distances > m2).float()
        #         correct_pairs += predictions.sum().item()
        #         total_pairs += batch_size

        # # Compute accuracy for contrastive evaluation
        # accuracy = 100 * correct_pairs / total_pairs
        # print(f'Contrastive Evaluation Accuracy: {accuracy:.2f}%')

        ######## Pretraining Phase (Embedding extraction, Contrastive training and Domain alignment)

        #Embedding extraction (Source)
        # Path to the saved source embeddings
        embedding_save_path = './model_ZNAX/save'+str(fold_idx)+'/cifar_embeddings.npy' # Replace with your actual path

        # Load the embeddings using numpy
        source_embeddings = np.load(embedding_save_path)

        # Convert the source embeddings into a PyTorch tensor
        source_embeddings = torch.tensor(source_embeddings, dtype=torch.float32)

        # Move the embeddings to the same device as the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        source_embeddings = source_embeddings.to(device)

        #print(f"Loaded source embeddings with shape: {source_embeddings.shape}")

        #Embedding extraction (Target)
        # Apply STFT and generate spectrograms
        def iq_to_spectrogram(iq_data, nperseg=256, noverlap=128):
            """Convert IQ (complex or stacked real/imag) to spectrogram using STFT."""
            iq_data = np.asarray(iq_data)
            if np.iscomplexobj(iq_data):
                complex_signal = iq_data
            else:
                iq_data = iq_data.reshape(-1)
                if iq_data.size < 2:
                    raise ValueError(f"iq_data too short: {iq_data.size}")
                half = iq_data.size // 2
                iq_data = iq_data[: 2 * half]
                I = iq_data[:half]
                Q = iq_data[half:]
                complex_signal = I + 1j * Q

            f, t, Zxx = stft(complex_signal, nperseg=nperseg, noverlap=noverlap)
            spectrogram = np.abs(Zxx)
            return spectrogram



        # 重要：为避免数据泄露，将训练数据分为三部分：
        # 1. 对比学习训练集 (60%)
        # 2. 对比学习验证集 (20%) 
        # 3. 监督学习训练集 (20%)
        # 测试集保持不变，用于最终评估
        
        # 使用相同的随机种子确保可重现
        # indices = np.random.RandomState(seed).permutation(len(train_data))
        
        # # 三分割：60% 对比训练, 20% 对比验证, 20% 监督训练
        # contrastive_train_size = int(0.6 * len(train_data))
        # contrastive_val_size = int(0.2 * len(train_data))
        # supervised_train_size = len(train_data) - contrastive_train_size - contrastive_val_size
        
        # contrastive_train_idx = indices[:contrastive_train_size]
        # contrastive_val_idx = indices[contrastive_train_size:contrastive_train_size + contrastive_val_size]
        # supervised_train_idx = indices[contrastive_train_size + contrastive_val_size:]
        
        # # 对比学习数据
        # pretrain_train_data = train_data[contrastive_train_idx]
        # pretrain_val_data = train_data[contrastive_val_idx]
        # pretrain_train_labels = train_labels[contrastive_train_idx]
        # pretrain_val_labels = train_labels[contrastive_val_idx]
        
        # # 监督学习数据（独立于对比学习数据）
        # supervised_train_data = train_data[supervised_train_idx]
        # supervised_train_labels = train_labels[supervised_train_idx]
        
        print(f"Data split - Contrastive train: {len(pretrain_train_data)}, Contrastive val: {len(pretrain_val_data)}, Supervised train: {len(supervised_train_data)}, Test: {len(test_data)}")

        # Data Transformations for pretraining
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])

        # Create Datasets for pretraining (using contrastive dataset)
        pretrain_train_dataset = ContrastiveH5Dataset(pretrain_train_data, pretrain_train_labels, transform=transform)
        pretrain_val_dataset = ContrastiveH5Dataset(pretrain_val_data, pretrain_val_labels, transform=transform)

        # Create Dataloaders for pretraining (修复CUDA错误)
        train_loader = DataLoader(pretrain_train_dataset, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(pretrain_val_dataset, batch_size=64, shuffle=False, num_workers=0)

        # Learnable margins

        class LearnableMargins(nn.Module):
            def __init__(self, m1_init=0.0, m2_init=0.6):
                """
                Initialize Learnable Margins
                Args:
                    m1_init (float): Initial value for margin m1
                    m2_init (float): Initial value for margin m2
                """
                super().__init__()
                
                # Learnable parameters for margins
                self.m1 = nn.Parameter(torch.tensor(m1_init, dtype=torch.float32))  # Margin m1
                self.m2 = nn.Parameter(torch.tensor(m2_init, dtype=torch.float32))  # Margin m2

            def forward(self):
                """
                Forward pass to return the learnable margins
                """
                return self.m1, self.m2

        # Dual Margin Contrastive Loss Function
        def dual_margin_contrastive_loss(emb1, emb2, label, m1, m2):
            """
            Dual Margin Contrastive Loss (DMCL) that computes the loss between two embeddings based on the margins.

            Args:
                emb1 (Tensor): Embedding from the first input sample
                emb2 (Tensor): Embedding from the second input sample
                label (Tensor): Pair label (1 for positive pair, 0 for negative pair)
                m1 (Tensor): Learnable margin for positive pairs
                m2 (Tensor): Learnable margin for negative pairs

            Returns:
                loss (Tensor): Computed contrastive loss
            """
            emb1 = F.normalize(emb1, p=2, dim=1)  
            emb2 = F.normalize(emb2, p=2, dim=1)  

            D = F.pairwise_distance(emb1, emb2)  # Intra-domain similarity

            #print(f"Positive pair distances: {D[label == 1]}")
            #print(f"Negative pair distances: {D[label == 0]}")
            #print(f"m2: {m2}, Distance D: {D}")

            label = label.view(-1, 1)  # Reshape label to match the distance tensor shape (batch_size, 1)

            #print(f"Positive pairs count: {torch.sum(label == 1).item()}")
            #print(f"Negative pairs count: {torch.sum(label == 0).item()}")

            # Positive pair loss: penalize when the distance is greater than margin m1
            loss_pos = label * torch.pow(torch.clamp(D - m1, min=0), 2)

            # Negative pair loss: penalize when the distance is less than margin m2
            #loss_neg = (1 - label) * torch.pow(torch.clamp(m2 - D + 1e-4, min=0), 2)
            if torch.sum(label == 0) > 0:  # Ensure there are negative pairs
                loss_neg = (1 - label) * torch.pow(torch.clamp(m2 - D + 1e-4, min=0), 2)
            else:
                loss_neg = torch.tensor(0.0, device=D.device)  # No contribution from negative loss

            # Final loss is a weighted sum of the positive and negative losses (weighting factor is optional)
            loss =  30 * torch.mean(loss_pos + loss_neg)

            return loss

        #Domain Alignment
        def mmd_loss(source, target, kernel='rbf', bandwidth=1.0, sample_size=1024):
            # Ensure embeddings are 2D (batch_size, emb_dim)
            source = source.view(source.size(0), -1)  # (N_source, D)
            target = target.view(target.size(0), -1)  # (batch_size, D)

            # Normalize embeddings for numerical stability
            source = F.normalize(source, p=2, dim=1)
            target = F.normalize(target, p=2, dim=1)

            # Subsample the source embeddings if necessary
            if source.size(0) > sample_size:
                idx = torch.randperm(source.size(0))[:sample_size]
                source = source[idx]

            # Compute pairwise squared distances
            def pairwise_distances(x, y):
                x_norm = (x ** 2).sum(dim=1, keepdim=True)  
                y_norm = (y ** 2).sum(dim=1, keepdim=True)  
                dist = x_norm + y_norm.T - 2.0 * torch.mm(x, y.T)
                return torch.clamp(dist, min=0.0)  

            # Compute pairwise RBF kernels
            def rbf_kernel(dist, gamma):
                return torch.exp(-gamma * dist)

            # Compute distances and kernels
            gamma = 1.0 / (2 * bandwidth ** 2)
            K_source = rbf_kernel(pairwise_distances(source, source), gamma)
            K_target = rbf_kernel(pairwise_distances(target, target), gamma)
            K_cross = rbf_kernel(pairwise_distances(source, target), gamma)

            # MMD loss
            loss = torch.mean(K_source) + torch.mean(K_target) - 2 * torch.mean(K_cross)
            return loss

        # Vision Transformer with learnable margins
        class VisionTransformerWithMargins(nn.Module):
            def __init__(self, pretrained_model, m1_init=0.0, m2_init=0.6):
                """
                Initialize Vision Transformer with learnable margins.

                Args:
                    pretrained_model (nn.Module): Pretrained Vision Transformer model
                    m1_init (float): Initial value for margin m1
                    m2_init (float): Initial value for margin m2
                """
                super().__init__()
                self.vit = pretrained_model  
                self.margins = LearnableMargins(m1_init, m2_init)  

            def forward(self, x):
                """
                Forward pass through the Vision Transformer.

                Args:
                    x (Tensor): Input data (spectrograms or other inputs)

                Returns:
                    Tensor: Output embeddings (features)
                """
                return self.vit(x)

            def get_margins(self):
                """
                Get the current learnable margins (m1, m2).

                Returns:
                    Tuple: Current values of m1 and m2
                """
                return self.margins()

        # Load Pretrained Model and Adjust Parameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load your pretrained model here (replace with actual model loading)
        vit = VisionTransformer(128, 8, 1, 128, 6, 8, 256)  
        state_dict = torch.load('./model_ZNAX/save'+str(fold_idx)+'/vit_model.pth')  # Replace with your actual model path

        # Adjust positional embeddings if needed
        if 'pos_embed' in state_dict:
            pos_embed_pretrained = state_dict['pos_embed']
            pos_embed_current = vit.pos_embed
            if pos_embed_pretrained.shape != pos_embed_current.shape:
                print("Interpolating positional embeddings...")
                num_patches_current = pos_embed_current.shape[1] - 1
                num_patches_pretrained = pos_embed_pretrained.shape[1] - 1
                cls_token_pretrained = pos_embed_pretrained[:, 0:1, :]
                patch_embeddings_pretrained = pos_embed_pretrained[:, 1:, :]
                patch_embeddings_pretrained = nn.functional.interpolate(
                    patch_embeddings_pretrained.permute(0, 2, 1).reshape(1, pos_embed_pretrained.shape[2], int(num_patches_pretrained**0.5), int(num_patches_pretrained**0.5)),
                    size=(int(num_patches_current**0.5), int(num_patches_current**0.5)),
                    mode='bicubic',
                    align_corners=False
                ).reshape(1, pos_embed_pretrained.shape[2], num_patches_current).permute(0, 2, 1)
                state_dict['pos_embed'] = torch.cat([cls_token_pretrained, patch_embeddings_pretrained], dim=1)

        # Adjust patch embedding weights if needed
        if 'patch_embed.projection.weight' in state_dict:
            projection_weight_pretrained = state_dict['patch_embed.projection.weight']
            projection_weight_current = vit.patch_embed.projection.weight
            if projection_weight_pretrained.shape[1] != projection_weight_current.shape[1]:
                print("Adjusting patch embedding weights for single-channel input...")
                state_dict['patch_embed.projection.weight'] = projection_weight_pretrained.mean(dim=1, keepdim=True)

        # Load the adjusted state_dict
        vit.load_state_dict(state_dict, strict=False)
        vit.to(device)


        m1_value = 0.0
        m2_value = 0.6

        # Model with learnable margins
        model = VisionTransformerWithMargins(vit, m1_value, m2_value).to(device)

        # Allow margins to be updated
        for param in model.parameters():
            param.requires_grad = True

        # Separate the learnable parameters from the model parameters
        base_params = [
            p for name, p in model.named_parameters() if "m1" not in name and "m2" not in name
        ]

        # Separate margin parameters (m1, m2) for optimization
        margin_params = [
            p for name, p in model.named_parameters() if "m1" in name or "m2" in name
        ]

        optimizer = AdamW(
            [{'params': base_params, 'lr': 1e-7},
            {'params': margin_params, 'lr': 1e-4}],
            weight_decay=1e-6
        )



        # Scheduler for the optimizer
        scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-10)

        projection_layer = nn.Linear(128, 64).to(device)
        #Pretraining loop ---->>> Contrastive training and domain alignment
        for epoch in range(30):  # Adjust number of epochs
            model.train()  
            total_train_loss = 0
            total_val_loss = 0
            batch_count = 0

            # Training Phase
            for img1, img2, pair_labels in train_loader:  
                img1, img2, pair_labels = img1.to(device), img2.to(device), pair_labels.to(device)  

                optimizer.zero_grad()  

                # Forward pass (get embeddings for both images)
                emb1 = F.normalize(model(img1), p=2, dim=1)  
                emb2 = F.normalize(model(img2), p=2, dim=1)  

                # Ensure both embeddings are projected to the same dimension (64)
                emb1 = projection_layer(emb1)  
                emb2 = projection_layer(emb2)  

                # source_embeddings is actually emb1 after projection in the training phase
                source_embeddings = emb1

                m1, m2 = model.get_margins()  # Get current margins (m1, m2)

                # Calculate contrastive loss
                contrastive_loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

                # Ensure the shapes of source_embeddings and emb2 are compatible for MMD loss
                assert source_embeddings.shape[1] == emb2.shape[1], f"Shape mismatch: {source_embeddings.shape[1]} vs {emb2.shape[1]}"
                assert source_embeddings.shape[0] == emb2.shape[0], f"Batch size mismatch: {source_embeddings.shape[0]} vs {emb2.shape[0]}"

                # Calculate MMD loss (source = source_embeddings, target = emb2)
                mmd_loss_val = mmd_loss(source_embeddings, emb2, sample_size=1024)  # Adjust sample_size as needed

                # Total loss is the sum of contrastive and MMD loss
                total_loss =  contrastive_loss + mmd_loss_val
                total_loss.backward()  # Backpropagate the loss

                #print(f"m2 gradient: {model.margins.m2.grad}")

                # # Inspect gradients for margin parameters (m2)
                # for name, param in model.named_parameters():
                #     if 'm2' in name:  # Check if it's the m2 parameter
                #         if param.grad is not None:
                #             print(f"Epoch {epoch+1}, Gradient for {name}: {param.grad.abs().mean().item()}")

                # Update parameters using the optimizer
                optimizer.step()

                # Accumulate the training loss
                total_train_loss += total_loss.item()
                batch_count += 1
                
                # Record batch-level metrics for LoRa contrastive training
                global_step_lora = epoch * len(train_loader) + batch_count
                writer.add_scalar('LoRa_Contrastive/Batch_Total_Loss', total_loss.item(), global_step_lora)
                writer.add_scalar('LoRa_Contrastive/Batch_Contrastive_Loss', contrastive_loss.item(), global_step_lora)
                writer.add_scalar('LoRa_Contrastive/Batch_MMD_Loss', mmd_loss_val.item(), global_step_lora)
                writer.add_scalar('LoRa_Contrastive/Margin_m1', m1.item(), global_step_lora)
                writer.add_scalar('LoRa_Contrastive/Margin_m2', m2.item(), global_step_lora)

            # Validation Phase
            model.eval()  
            with torch.no_grad():  
                for img1, img2, pair_labels in val_loader:
                    img1, img2, pair_labels = img1.to(device), img2.to(device), pair_labels.to(device)

                    
                    emb1 = F.normalize(model(img1), p=2, dim=1)  # Source embedding for img1
                    emb2 = F.normalize(model(img2), p=2, dim=1)  # Target embedding for img2

                    
                    emb1 = projection_layer(emb1)  
                    emb2 = projection_layer(emb2)  

                    # source_embeddings is actually emb1 after projection in the validation phase
                    source_embeddings = emb1

                    m1, m2 = model.get_margins()  # Get current margins

                    # Ensure the shapes of source_embeddings and emb2 are compatible for MMD loss
                    assert source_embeddings.shape[1] == emb2.shape[1], f"Shape mismatch: {source_embeddings.shape[1]} vs {emb2.shape[1]}"
                    assert source_embeddings.shape[0] == emb2.shape[0], f"Batch size mismatch: {source_embeddings.shape[0]} vs {emb2.shape[0]}"

                    # Calculate validation contrastive loss
                    val_contrastive_loss = dual_margin_contrastive_loss(emb1, emb2, pair_labels, m1, m2)

                    # Calculate validation MMD loss
                    val_mmd_loss = mmd_loss(source_embeddings, emb2, sample_size=1024)  # Use the fixed source_embeddings

                    # Accumulate the validation loss
                    total_val_loss += val_contrastive_loss.item() + val_mmd_loss.item()

            
            # Calculate average losses
            avg_train_loss_lora = total_train_loss / len(train_loader)
            avg_val_loss_lora = total_val_loss / len(val_loader)
            
            # Record epoch-level metrics for LoRa contrastive training
            writer.add_scalar('LoRa_Contrastive/Epoch_Train_Loss', avg_train_loss_lora, epoch)
            writer.add_scalar('LoRa_Contrastive/Epoch_Val_Loss', avg_val_loss_lora, epoch)
            writer.add_scalar('LoRa_Contrastive/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Epoch [{epoch+1}/40], "
                f"Train Loss: {avg_train_loss_lora:.4f}, "
                f"Val Loss: {avg_val_loss_lora:.4f}, "
                f"m1: {m1.item():.4f}, m2: {m2.item():.4f}")

            
            scheduler.step()


        # Specify the save path for the model
        save_path = './model_ZNAX/save'+str(fold_idx)+'/vit_model_finetuned.pth' #Replace with your actual path

        # Save the model's state_dict
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    
    
    #### Supervised Fine-tuning Phase
    print("\n" + "="*50)
    print("Phase 2: Supervised Fine-tuning")
    print("="*50)

    # 使用相同的统一数据划分（无需重新加载和处理数据）
    # train_data, test_data, train_labels, test_labels 已经通过5折交叉验证划分好了
    # Data Transformations for supervised fine-tuning (使用相同的mean和std)
    finetune_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[mean], std=[std])  # 使用相同的归一化参数
    ])

    # Create Datasets using separated data for supervised fine-tuning (避免数据泄露)
    print(f"Creating datasets...")
    print(f"Supervised train data shape: {supervised_train_data.shape}")
    print(f"Supervised train labels shape: {supervised_train_labels.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    # Split supervised_train into train/val to avoid using test for model selection
    # (test is reserved ONLY for final reporting)
    sup_x_train, sup_x_val, sup_y_train, sup_y_val = train_test_split(
        supervised_train_data,
        supervised_train_labels,
        test_size=0.2,
        random_state=seed,
        stratify=supervised_train_labels if len(np.unique(supervised_train_labels)) > 1 else None,
    )

    finetune_train_dataset = SupervisedH5Dataset(sup_x_train, sup_y_train, transform=finetune_transform)
    finetune_val_dataset = SupervisedH5Dataset(sup_x_val, sup_y_val, transform=finetune_transform)
    finetune_test_dataset = SupervisedH5Dataset(test_data, test_labels, transform=finetune_transform)

    # Create Dataloaders for supervised fine-tuning (修复CUDA错误)
    finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=64, shuffle=True, num_workers=0)
    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size=64, shuffle=False, num_workers=0)
    finetune_test_loader = DataLoader(finetune_test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(
        f"Fine-tuning - Train samples: {len(finetune_train_dataset)}, "
        f"Val samples: {len(finetune_val_dataset)}, "
        f"Test samples: {len(finetune_test_dataset)}"
    )
    print("NOTE: Val split is from supervised train only; test is untouched")


    # Vision Transformer and Classifier
    class VisionTransformer(nn.Module):
        def __init__(self, img_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim):
            super().__init__()
            self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.randn(1, (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) + 1, embed_dim))
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



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Adjust ViT for Spectrogram Input
    spectrogram_size = (256, 65)  

    patch_size = (16, 8)

    vit = VisionTransformer(spectrogram_size, patch_size, 1, 128, 6, 8, 256)


    # Load pretrained model
    state_dict = torch.load('./model_ZNAX/save'+str(fold_idx)+'/vit_model_finetuned.pth', map_location=device) #Replace with your actual path

    # Positional Embeddings Adjustments (if required)
    if 'pos_embed' in state_dict:
        pos_embed_pretrained = state_dict['pos_embed']
        pos_embed_current = vit.pos_embed
        if pos_embed_pretrained.shape != pos_embed_current.shape:
            print("Interpolating positional embeddings...")
            num_patches_current = pos_embed_current.shape[1] - 1
            num_patches_pretrained = pos_embed_pretrained.shape[1] - 1
            cls_token_pretrained = pos_embed_pretrained[:, 0:1, :]
            patch_embeddings_pretrained = pos_embed_pretrained[:, 1:, :]
            patch_embeddings_pretrained = nn.functional.interpolate(
                patch_embeddings_pretrained.permute(0, 2, 1).reshape(1, pos_embed_pretrained.shape[2], int(num_patches_pretrained**0.5), int(num_patches_pretrained**0.5)),
                size=(int(num_patches_current**0.5), int(num_patches_current**0.5)),
                mode='bicubic',
                align_corners=False
            ).reshape(1, pos_embed_pretrained.shape[2], num_patches_current).permute(0, 2, 1)
            state_dict['pos_embed'] = torch.cat([cls_token_pretrained, patch_embeddings_pretrained], dim=1)

    # Load adjusted weights
    vit.load_state_dict(state_dict, strict=False)
    vit.to(device)

    # Attach Classifier
    num_classes = len(np.unique(supervised_train_labels))  # 类别数基于监督训练标签
    model = VisionTransformerWithClassifier(vit, num_classes).to(device)
    
    print(f"Number of classes for supervised learning: {num_classes}")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # Supervised fine-tuning loop
    num_epochs = 100

    def train_finetune(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, writer, save_dir, fold_idx=0, days='3,4'):
        """Train and validate the supervised fine-tuning, save best and final checkpoints.

        Args:
            model: model to train
            train_loader: training DataLoader
            val_loader: validation DataLoader (used for model selection)
            criterion: loss function
            optimizer: optimizer
            scheduler: LR scheduler
            device: torch device
            num_epochs: number of epochs
            writer: tensorboard writer
            save_dir: directory to save checkpoints
            fold_idx: fold index used in filenames
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = -1.0
        best_ckpt_path = os.path.join(save_dir, f'best_finetune_fold{fold_idx}_{days}.pth')
        final_ckpt_path = os.path.join(save_dir, f'final_finetune_fold{fold_idx}_{days}.pth')

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device).float()
                targets = targets.to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate training stats
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                # Record batch-level metrics
                global_step_finetune = epoch * len(train_loader) + batch_idx + 1
                batch_acc = (predicted == targets).float().mean().item()
                writer.add_scalar('Supervised_Finetune/Batch_Loss', loss.item(), global_step_finetune)
                writer.add_scalar('Supervised_Finetune/Batch_Accuracy', batch_acc, global_step_finetune)

            scheduler.step()

            # Calculate epoch training metrics
            epoch_loss = running_loss / max(total_samples, 1)
            epoch_accuracy = correct_predictions / max(total_samples, 1)

            # Validation pass (compute accuracy on val_loader)
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_inputs = v_inputs.to(device).float()
                    v_targets = v_targets.to(device).long()
                    v_outputs = model(v_inputs)
                    _, v_pred = v_outputs.max(1)
                    val_correct += (v_pred == v_targets).sum().item()
                    val_total += v_targets.size(0)

            val_acc = 100.0 * val_correct / max(val_total, 1)

            # Record epoch-level metrics
            writer.add_scalar('Supervised_Finetune/Epoch_Loss', epoch_loss, epoch)
            writer.add_scalar('Supervised_Finetune/Epoch_Accuracy', epoch_accuracy, epoch)
            writer.add_scalar('Supervised_Finetune/Val_Accuracy', val_acc, epoch)
            writer.add_scalar('Supervised_Finetune/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            # Print statistics
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model by validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'val_acc': float(val_acc),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, best_ckpt_path)
                print(f"Saved best finetune checkpoint: {best_ckpt_path} (Val Acc: {best_val_acc:.2f}%)")

        # Save final model checkpoint after all epochs
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': float(best_val_acc),
        }, final_ckpt_path)
        print(f"Saved final finetune checkpoint: {final_ckpt_path}")

    # Call the fine-tuning function with the correct data loader and validation loader
    save_dir = './model_ZNAX/save'+str(fold_idx)
    train_finetune(model, finetune_train_loader, finetune_val_loader, criterion, optimizer, scheduler, device, num_epochs, writer, save_dir, fold_idx, days=days)

    # Final evaluation on test data (Emitter identification) 
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, targets in finetune_test_loader:  # 使用正确的测试数据加载器
            images, targets = images.to(device), targets.to(device).long()
            outputs = model(images.float())
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(targets).sum().item()
            test_total += targets.size(0)

    test_acc = 100. * test_correct / test_total
    print(f"Fold {fold_idx+1} - Final Test Accuracy: {test_acc:.2f}%")
    
    # Record final test accuracy
    writer.add_scalar('Final_Results/Test_Accuracy', test_acc, 0)
    writer.add_text('Final_Results/Summary', f'Fold {fold_idx+1} - Test Accuracy: {test_acc:.2f}%', 0)
    
    # Close TensorBoard writer
    writer.close()
    
    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5-Fold Cross Validation for LoRa RF Fingerprinting')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--fold_idx', type=int, default=3, help='specific fold to run (0-4), -1 for all folds')
    parser.add_argument('--gpus', type=str, default="1", help='GPU devices to use')
    parser.add_argument('--port', type=str, default="12347", help='port number')
    parser.add_argument('--days', type=str, default='4', help='which day data to use (1, 2, or 3)')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    if args.fold_idx >= 0:
        # 运行指定的fold
        print(f"Running single fold: {args.fold_idx}")
        accuracy = main(fold_idx=args.fold_idx, seed=args.seed, days=args.days)
        print(f"Final accuracy for fold {args.fold_idx}: {accuracy:.2f}%")
    else:
        # 运行5折交叉验证
        print("Running 5-Fold Cross Validation...")
        print("=" * 60)
        
        all_accuracies = []
        
        for fold in range(5):
            print(f"\n{'='*20} FOLD {fold+1}/5 {'='*20}")
            # try:
            accuracy = main(fold_idx=fold, seed=args.seed, days=args.days)
            all_accuracies.append(accuracy)
            print(f"Fold {fold+1} completed with accuracy: {accuracy:.2f}%")
            # except Exception as e:
                # print(f"Error in fold {fold+1}: {str(e)}")
                # continue
        
        # 计算统计结果
        if all_accuracies:
            mean_accuracy = np.mean(all_accuracies)
            std_accuracy = np.std(all_accuracies)
            
            print("\n" + "="*60)
            print("5-FOLD CROSS VALIDATION RESULTS")
            print("="*60)
            print(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in all_accuracies]}")
            print(f"Mean Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
            print(f"Min Accuracy: {min(all_accuracies):.2f}%")
            print(f"Max Accuracy: {max(all_accuracies):.2f}%")
            print("="*60)
            
            # 保存结果到文件
            results = {
                'fold_accuracies': all_accuracies,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'min_accuracy': min(all_accuracies),
                'max_accuracy': max(all_accuracies),
                'seed': args.seed
            }
            
            import json
            with open(f'./model_ZNAX/save/5fold_results_seed{args.seed}_{args.days}.json', 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to: ./model_ZNAX/save/5fold_results_seed{args.seed}_{args.days}.json")
        else:
            print("No successful folds completed!")