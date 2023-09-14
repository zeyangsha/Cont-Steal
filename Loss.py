import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_target, emb_surrogate):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        representations = torch.cat([emb_target, emb_surrogate], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask.to(device) * torch.exp(similarity_matrix / self.temperature.to(device))
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = (torch.sum(loss_partial)) / (2 * self.batch_size)
        return loss.mean()
