import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(512, 90)  # Assuming 90 classes, adjust accordingly

    def forward(self, images):
        with torch.no_grad():
            features = self.clip_model.encode_image(images)
        return self.classifier(features.float())
    

def info_nce_loss(features):
    device = features.device
    batch_size = features.shape[0]
    labels = torch.cat([torch.arange(batch_size // 2) for i in range(2)], dim=0).to(device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

    logits = logits / 0.2
    return logits, labels