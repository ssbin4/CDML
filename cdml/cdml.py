import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import googlenet
from . import loss


class TripletBase(nn.Module):
    def __init__(self, embedding_size=128, n_class=99, pretrained=False):
        super(TripletBase, self).__init__()
        n_mid = 1024
        self.googlenet = googlenet.googlenet(pretrained=pretrained)
        self.bn1 = nn.BatchNorm1d(n_mid)
        self.fc1 = nn.Linear(n_mid, embedding_size)
        self.loss_fn = loss.TripletLoss()

    def forward(self, x, use_loss=True):
        embedding_y_orig = self.googlenet(x)
        embedding = self.bn1(embedding_y_orig)
        embedding_z = self.fc1(embedding)
        if use_loss:
            jm = self.loss_fn(embedding_z)
            return jm, embedding_y_orig, embedding_z
        return embedding_z

class TripletFC(nn.Module):
    def __init__(self, embedding_size=128, n_class=99, pretrained=False):
        super(TripletFC, self).__init__()
        n_mid = 1024
        self.embedding_size = embedding_size
        self.embedding = TripletBase(embedding_size, n_class, pretrained)
        self.softmax_classifier = nn.Linear(embedding_size, n_class)

    def forward(self, x, t):
        label = t.squeeze(-1)
        jm, embedding_y, embedding_z = self.embedding(x)
        logits_orig = self.softmax_classifier(embedding_z)
        jclass = F.nll_loss(F.log_softmax(logits_orig, dim=1), label, reduction='mean')

        return jclass, embedding_y, embedding_z


class TripletCDML(nn.Module):
    def __init__(self, embedding_size=128, n_class=99, n_cand=4,
                 alpha=0.1, beta=1,
                 l2_norm=False, pretrained=False,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 batch=60):
        super(TripletCDML, self).__init__()
        n_mid = 1024
        self.alpha = alpha
        self.beta = beta
        self.n_cand = n_cand
        self.l2_norm = l2_norm
        self.embedding_size = embedding_size
        self.loss_fn = loss.TripletLoss()
        self.embedding = TripletBase(embedding_size, n_class, pretrained)
        self.softmax_classifier = nn.Linear(embedding_size, n_class)
        self.device = device
        self.batch_size = batch
        self.triplet_size = int(batch/3)

    def generateCandidates(self, x, t):
        anchor, positive, negative = torch.chunk(x, 3, dim=0)
        anchor_label, positive_label, negative_label = torch.chunk(t, 3, dim=0)
        total_length = float(self.n_cand + 1)
        candidates = negative.clone().detach()
        candidates = candidates.unsqueeze(dim=0)
        labels = negative_label.clone().detach()
        mid = (anchor + negative)/2
        for n_idx in range(self.n_cand):
            left_length = float(n_idx + 1)
            right_length = total_length - left_length
            inner_pts = (negative * left_length + mid * right_length) / total_length
            inner_pts = inner_pts.unsqueeze(dim=0)
            if self.l2_norm:
                inner_pts = F.L2Normalization(inner_pts)
            candidates = torch.cat([candidates, inner_pts], dim=0)
            labels = torch.cat([labels, negative_label], dim=0)
        return labels, candidates

    def pickHardestSample(self, cand, loss):
        cand = cand.view(-1, self.triplet_size, self.embedding_size)
        loss = loss.view(-1, self.triplet_size)
        mask = torch.le(loss, self.alpha)
        less = mask * loss
        indices = torch.argmax(less, dim=0)
        cand = cand.view(-1, self.embedding_size)
        indices = self.triplet_size*indices + torch.arange(self.triplet_size).to(self.device)
        hardest = torch.index_select(cand, 0, indices)
        return hardest

    def forward(self, x, t, jclass):
        coeff = np.exp(-self.beta / jclass)

        label = t.squeeze(-1)
        jm, embedding_y, embedding_z = self.embedding(x)
        logits_orig = self.softmax_classifier(embedding_z)
        jclass = F.nll_loss(F.log_softmax(logits_orig, dim=1), label, reduction='mean')

        cand_label, candidates = self.generateCandidates(embedding_z, label)
        candidates = candidates.view(-1, self.embedding_size)

        logits_cand = self.softmax_classifier(candidates)

        cand_loss = F.nll_loss(F.log_softmax(logits_cand, dim=1), cand_label, reduction='none')
        jcand = F.nll_loss(F.log_softmax(logits_cand, dim=1), cand_label, reduction='mean')

        hard_samples = self.pickHardestSample(candidates, cand_loss)
        anchor, positive, negative = torch.chunk(embedding_z, 3, dim=0)
        syn_samples = torch.cat([anchor, positive, hard_samples], dim=0)

        jsyn = self.loss_fn(syn_samples)
        jm = coeff * jm
        jsyn = (1 - coeff) * jsyn
        jmetric = jm + jsyn
        return jmetric, jclass, jcand, embedding_z