"""
Prototypical Network for few-shot tactical strategy classification.

Reference:
    - Snell et al., 2017: "Prototypical Networks for Few-shot Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PrototypicalNetwork(nn.Module):
    """
    Few-shot classifier using class prototypes in embedding space.

    Given an encoder that maps inputs to embeddings, computes class
    prototypes from support examples and classifies queries by
    distance to nearest prototype.
    """

    def __init__(self, encoder, distance="euclidean"):
        """
        Args:
            encoder: backbone network mapping (B, C, T, V) → (B, D)
            distance: "euclidean" or "cosine"
        """
        super().__init__()
        self.encoder = encoder
        self.distance = distance

    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """
        Compute class prototypes as mean embedding per class.

        Args:
            support_embeddings: (n_support, D)
            support_labels: (n_support,) integer class labels
            n_way: number of classes

        Returns:
            prototypes: (n_way, D)
        """
        prototypes = torch.zeros(n_way, support_embeddings.shape[1],
                                 device=support_embeddings.device)
        for c in range(n_way):
            mask = support_labels == c
            if mask.sum() > 0:
                prototypes[c] = support_embeddings[mask].mean(dim=0)
        return prototypes

    def compute_distances(self, queries, prototypes):
        """
        Compute distances from queries to each prototype.

        Args:
            queries: (n_query, D)
            prototypes: (n_way, D)

        Returns:
            distances: (n_query, n_way) — lower = closer
        """
        if self.distance == "euclidean":
            # (n_query, 1, D) - (1, n_way, D) → (n_query, n_way)
            return torch.cdist(queries, prototypes)
        elif self.distance == "cosine":
            # Cosine distance = 1 - cosine_similarity
            queries_norm = F.normalize(queries, dim=1)
            protos_norm = F.normalize(prototypes, dim=1)
            similarity = torch.mm(queries_norm, protos_norm.t())
            return 1.0 - similarity
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

    def forward(self, support_x, support_y, query_x, n_way):
        """
        Full forward pass for one episode.

        Args:
            support_x: (n_support, C, T, V) support skeleton sequences
            support_y: (n_support,) support labels
            query_x: (n_query, C, T, V) query skeleton sequences
            n_way: number of classes

        Returns:
            log_probs: (n_query, n_way) log-probabilities
            prototypes: (n_way, D) for visualization
        """
        # Encode all examples
        support_emb = self.encoder(support_x)  # (n_support, D)
        query_emb = self.encoder(query_x)      # (n_query, D)

        # Compute prototypes
        prototypes = self.compute_prototypes(support_emb, support_y, n_way)

        # Compute distances and convert to log-probabilities
        distances = self.compute_distances(query_emb, prototypes)
        log_probs = F.log_softmax(-distances, dim=1)

        return log_probs, prototypes

    def predict(self, query_x, prototypes):
        """
        Predict using pre-computed prototypes (inference mode).

        Args:
            query_x: (n_query, C, T, V)
            prototypes: (n_way, D)

        Returns:
            predictions: (n_query,) predicted class indices
            confidences: (n_query,) confidence scores
            margins: (n_query,) margin between top-2 predictions
        """
        query_emb = self.encoder(query_x)
        distances = self.compute_distances(query_emb, prototypes)

        # Softmax probabilities
        probs = F.softmax(-distances, dim=1)
        predictions = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

        # Margin: difference between best and second-best
        sorted_dists, _ = distances.sort(dim=1)
        margins = sorted_dists[:, 1] - sorted_dists[:, 0]

        return predictions, confidences, margins

    @staticmethod
    def proto_loss(log_probs, query_labels):
        """
        Prototypical network loss (negative log-likelihood).

        Args:
            log_probs: (n_query, n_way)
            query_labels: (n_query,)

        Returns:
            loss: scalar
            accuracy: scalar
        """
        loss = F.nll_loss(log_probs, query_labels)
        preds = log_probs.argmax(dim=1)
        accuracy = (preds == query_labels).float().mean()
        return loss, accuracy
