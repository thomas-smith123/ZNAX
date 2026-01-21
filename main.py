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

import warnings
warnings.filterwarnings("ignore")

# Mount Google Drive / Local Drive
# drive.mount('/content/drive') #Replace with your actual path

# Patch Embedding

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
for epoch in range(30):
    model.train()
    total_train_loss = 0
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
    print(
        f"Epoch [{epoch+1}/30], Train Loss: {total_train_loss / len(train_loader):.4f}, "
        f"Val Loss: {total_val_loss / len(val_loader):.4f}, "
        f"m1: {m1.item():.4f}, m2: {m2.item():.4f}"
    )

# After training, save embeddings and model


cifar_embeddings = np.array(cifar_embeddings)  # Convert the list of embeddings to a numpy array

embedding_save_path = './model_ZNAX/save/cifar_embeddings.npy'  # Path to save embeddings --- Replace with your actual path
np.save(embedding_save_path, cifar_embeddings)
print(f"Embeddings saved to {embedding_save_path}")

# Specify the save path for the model
save_path = './model_ZNAX/save/vit_model.pth' #Define path to save---Replace with your actual path

# Save the model's state_dict
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")


#Evaluation on Test Data
model.eval()  
total_test_loss = 0

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        batch_size = test_images.size(0)

        # Create pairs within the test batch
        idx = torch.randperm(batch_size)
        test_img1, test_img2 = test_images, test_images[idx]
        test_pair_labels = (test_labels == test_labels[idx]).float().to(device)

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

save_path = './model_ZNAX/save/vit_model.pth' # Replace with your actual path
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
embedding_save_path = './model_ZNAX/save/cifar_embeddings.npy' # Replace with your actual path

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
    """
    Convert raw IQ data to spectrogram using STFT.
    Args:
        iq_data (numpy.ndarray): The raw IQ signal (real and imaginary concatenated).
        nperseg (int): Length of each segment for STFT.
        noverlap (int): Number of overlapping samples between segments.

    Returns:
        spectrogram (numpy.ndarray): The time-frequency spectrogram.
    """
    # Split the real (I) and imaginary (Q) parts from concatenated IQ data
    I = iq_data[::2]  # Real part (even indices)
    Q = iq_data[1::2]  # Imaginary part (odd indices)

    # Combine I and Q to form complex signal
    complex_signal = I + 1j * Q

    # Apply STFT
    f, t, Zxx = stft(complex_signal, nperseg=nperseg, noverlap=noverlap)

    # Convert to magnitude spectrogram
    spectrogram = np.abs(Zxx)

    return spectrogram


class CustomH5Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data.astype(np.float32)
        self.labels = labels.flatten()  # Flatten the labels to make it a 1D array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
            neg_idx = idx
            while neg_idx == idx:  # Ensure it's a different sample
                neg_idx = random.randint(0, len(self.data) - 1)

            sample_neg = self.data[neg_idx]
            sample_neg = self.transform(sample_neg)
            pair_label = torch.tensor(0.0 if label != self.labels[neg_idx] else 1.0)

            return sample, sample2, pair_label  # Positive pair (same class)

        return sample, label





# RFFI dataset
dataset_path = './dataset/dataset_training_no_aug.h5' # Replace with your actual path
with h5py.File(dataset_path, 'r') as f:
    data = np.array(f['data'])    
    labels = np.array(f['label']) 

# Apply STFT to convert IQ samples to spectrograms
spectrograms = []
for iq_sample in data:
    spectrogram = iq_to_spectrogram(iq_sample, nperseg=256, noverlap=128)
    spectrograms.append(spectrogram)

# Convert list of spectrograms to numpy array
spectrograms = np.array(spectrograms)

# Debug_Check the shape of the spectrograms
print(f"Shape of spectrograms: {spectrograms.shape}")

# Reshape Data to match the expected input format: 
spectrograms = spectrograms.reshape(-1, 256, 65, 1)  
labels = labels.flatten() - 1  

mean = np.mean(spectrograms)
std = np.std(spectrograms)


train_size = int(0.8 * len(spectrograms))  
val_size = len(spectrograms) - train_size  
train_data, val_data = random_split(spectrograms, [train_size, val_size])
train_labels, val_labels = random_split(labels, [train_size, val_size])

# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

# Create Datasets
train_dataset = CustomH5Dataset(train_data.dataset, train_labels.dataset, transform=transform)
val_dataset = CustomH5Dataset(val_data.dataset, val_labels.dataset, transform=transform)

# Create Dataloaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

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
state_dict = torch.load('./model_ZNAX/save/vit_model.pth')  # Replace with your actual model path

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
for epoch in range(40):  # Adjust number of epochs
    model.train()  
    total_train_loss = 0
    total_val_loss = 0

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

    
    print(f"Epoch [{epoch+1}/40], "
          f"Train Loss: {total_train_loss/len(train_loader):.4f}, "
          f"Val Loss: {total_val_loss/len(val_loader):.4f}, "
          f"m1: {m1.item():.4f}, m2: {m2.item():.4f}")

    
    scheduler.step()


# Specify the save path for the model
save_path = './model_ZNAX/save/vit_model_finetuned.pth' #Replace with your actual path

# Save the model's state_dict
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

#### Supervised Fine-tuning Phase

#Preprocessing
# Function to compute STFT and generate spectrogram
def iq_to_spectrogram(iq_data, nperseg=256, noverlap=128):
    """
    Convert raw IQ data to spectrogram using STFT.
    Args:
        iq_data (numpy.ndarray): The raw IQ signal (real and imaginary concatenated).
        nperseg (int): Length of each segment for STFT.
        noverlap (int): Number of overlapping samples between segments.

    Returns:
        spectrogram (numpy.ndarray): The time-frequency spectrogram.
    """
    # Split the real (I) and imaginary (Q) parts from concatenated IQ data
    I = iq_data[::2]  # Real part (even indices)
    Q = iq_data[1::2]  # Imaginary part (odd indices)

    # Combine I and Q to form complex signal
    complex_signal = I + 1j * Q

    # Apply STFT
    f, t, Zxx = stft(complex_signal, nperseg=nperseg, noverlap=noverlap)

    # Convert to magnitude spectrogram
    spectrogram = np.abs(Zxx)

    return spectrogram


# Custom Dataset Class
class CustomH5Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data.astype(np.float32)
        self.labels = labels.flatten()  # Flatten labels for consistency
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# RFFI dataset
dataset_path = './dataset/dataset_training_no_aug.h5' #Replace with your actual path
with h5py.File(dataset_path, 'r') as f:
    data = np.array(f['data'])    
    labels = np.array(f['label']) 

# Apply STFT to convert IQ samples to spectrograms
spectrograms = []
for iq_sample in data:
    spectrogram = iq_to_spectrogram(iq_sample, nperseg=256, noverlap=128)
    spectrograms.append(spectrogram)


spectrograms = np.array(spectrograms)

# Debug: Check the shape of the spectrograms
#print(f"Shape of spectrograms: {spectrograms.shape}")

# Reshape Data to match the expected input format: (15000, 256, 65, 1)
spectrograms = spectrograms.reshape(-1, 256, 65, 1)  
labels = labels.flatten() - 1  # Flatten and ensure 0-based indexing

# Calculate the mean and std of the spectrograms across the whole dataset
mean = np.mean(spectrograms)
std = np.std(spectrograms)

#print(f"Spectrograms Mean: {mean}, Std: {std}")

# Train-Test Split: 90% for training, 10% for testing
train_data, test_data, train_labels, test_labels = train_test_split(
    spectrograms, labels, test_size=0.1, random_state=42
)

# Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[mean], std=[std])  # Normalize based on calculated mean and std
])

# Create Datasets
train_dataset = CustomH5Dataset(train_data, train_labels, transform=transform)
test_dataset = CustomH5Dataset(test_data, test_labels, transform=transform)

# Create Dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)


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
state_dict = torch.load('./model_ZNAX/save/vit_model_finetuned.pth', map_location=device) #Replace with your actual path

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
num_classes = len(np.unique(labels))
model = VisionTransformerWithClassifier(vit, num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=1e-5) 

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

#Supervised fine-tuning loop

num_epochs = 55

def train(model, train_loader, criterion, optimizer, scheduler, device, num_epochs):
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

            # Calculate loss and identification accuracy
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        
        scheduler.step()



        # Calculate epoch loss and identification accuracy
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        # Print statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {epoch_loss:.4f}, "
              f"Accuracy: {epoch_accuracy:.4f}")

# Call the training function
train(model, train_loader, criterion, optimizer, scheduler, device, num_epochs)

# Final evaluation on test data (Emitter identification) 
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device).long()
        outputs = model(images.float())
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(targets).sum().item()
        test_total += targets.size(0)

test_acc = 100. * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")