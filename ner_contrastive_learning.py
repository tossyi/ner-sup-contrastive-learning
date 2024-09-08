import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class NERDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ContrastiveModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


def contrastive_loss(output1, output2, label, margin=1.0):
    """
    Computes the contrastive loss between two output embeddings.

    Parameters:
    - output1: The output embeddings from the first input.
    - output2: The output embeddings from the second input.
    - label: A tensor indicating whether the pairs are similar (1) or dissimilar (0).
    - margin: The margin for dissimilar pairs, default is 1.0.

    Returns:
    - loss: The computed contrastive loss value.
    """
    # Calculate the pairwise distance between the two outputs
    distance = nn.functional.pairwise_distance(output1, output2)

    # Compute the contrastive loss
    loss = torch.mean(
        (1 - label) * torch.pow(distance, 2)  # Loss for dissimilar pairs
        + (label)
        * torch.pow(
            torch.clamp(margin - distance, min=0.0), 2
        )  # Loss for similar pairs
    )

    return loss


# データの準備
# Ontonotes5 dataset preparation
from datasets import load_dataset

# Load the Ontonotes5 dataset
dataset = load_dataset("tner/ontonotes5")

# Extract texts and labels from the dataset
texts = [item['tokens'] for item in dataset['train']]  # Assuming 'train' split
labels = [item['ner_tags'] for item in dataset['train']]  # Assuming 'ner_tags' contains the labels

print(texts[0])
print(labels[0])
exit

dataset = NERDataset(texts, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# モデルの初期化
model = ContrastiveModel(input_dim=768, output_dim=128)  # 例: BERTの出力次元
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ
for epoch in range(1):
    for text_batch, label_batch in dataloader:
        optimizer.zero_grad()
        output1 = model(text_batch[0])  # 1つ目の入力
        output2 = model(text_batch[1])  # 2つ目の入力
        loss = contrastive_loss(output1, output2, label_batch)
        loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
        print(f'Model saved after epoch {epoch}')
