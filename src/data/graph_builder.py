"""
Dual-player spatio-temporal graph construction.

Builds a unified graph with 34 nodes (17 joints x 2 players) with:
    - Intra-player edges (COCO skeleton topology)
    - Inter-player edges (corresponding joints between players)
    - Temporal edges (self-connections across frames)

The adjacency matrix is partitioned into 3 subsets for ST-GCN:
    1. Self-connections (identity)
    2. Centripetal edges (toward center of gravity)
    3. Centrifugal edges (away from center of gravity)
"""
import numpy as np
import torch
from ..config import (
    NUM_JOINTS, NUM_PLAYERS, NUM_NODES,
    COCO_SKELETON, INTER_PLAYER_EDGES,
)


class GraphBuilder:
    """Builds adjacency matrices for the dual-player skeleton graph."""

    def __init__(self, use_inter_player=True, single_player=False):
        """
        Args:
            use_inter_player: include cross-player edges
            single_player: use only player 1 (17 nodes instead of 34)
        """
        self.use_inter_player = use_inter_player
        self.single_player = single_player
        self.num_nodes = NUM_JOINTS if single_player else NUM_NODES

    def build_adjacency(self):
        """
        Build the partitioned adjacency matrix.

        Returns:
            adjacency: (3, V, V) tensor — 3 partitions for ST-GCN
        """
        V = self.num_nodes
        A = np.zeros((V, V), dtype=np.float32)

        # Intra-player edges for player 1
        for (i, j) in COCO_SKELETON:
            A[i, j] = 1
            A[j, i] = 1

        if not self.single_player:
            # Intra-player edges for player 2 (offset by NUM_JOINTS)
            for (i, j) in COCO_SKELETON:
                A[i + NUM_JOINTS, j + NUM_JOINTS] = 1
                A[j + NUM_JOINTS, i + NUM_JOINTS] = 1

            # Inter-player edges
            if self.use_inter_player:
                for (i, j) in INTER_PLAYER_EDGES:
                    A[i, j] = 1
                    A[j, i] = 1

        # Self-connections
        I = np.eye(V, dtype=np.float32)

        # Partition into 3 subsets following Yan et al. 2018:
        # 1) Identity (self-loops)
        # 2) Centripetal (neighbors closer to root)
        # 3) Centrifugal (neighbors farther from root)
        A_partitioned = self._partition_adjacency(A, I)

        return torch.tensor(A_partitioned)

    def _partition_adjacency(self, A, I):
        """
        Partition adjacency into identity + inward + outward subsets.

        Uses distance from root joint (joint 0 = nose) to determine
        centripetal vs centrifugal direction.
        """
        V = A.shape[0]

        # Compute distance from root for each joint
        root = 0
        distances = self._bfs_distances(A, root, V)

        A_inward = np.zeros_like(A)
        A_outward = np.zeros_like(A)

        for i in range(V):
            for j in range(V):
                if A[i, j] == 0:
                    continue
                if distances[j] < distances[i]:
                    # j is closer to root → inward (centripetal)
                    A_inward[i, j] = 1
                else:
                    # j is farther or equal → outward (centrifugal)
                    A_outward[i, j] = 1

        # Normalize each partition
        A_identity = self._normalize_adj(I)
        A_inward = self._normalize_adj(A_inward)
        A_outward = self._normalize_adj(A_outward)

        return np.stack([A_identity, A_inward, A_outward], axis=0)

    @staticmethod
    def _bfs_distances(A, root, V):
        """BFS shortest path distances from root."""
        distances = np.full(V, float("inf"))
        distances[root] = 0
        queue = [root]
        visited = {root}

        while queue:
            node = queue.pop(0)
            for neighbor in range(V):
                if A[node, neighbor] > 0 and neighbor not in visited:
                    distances[neighbor] = distances[node] + 1
                    visited.add(neighbor)
                    queue.append(neighbor)

        # Unreachable nodes (e.g., player 2 from player 1 root if no inter-edges)
        # get max distance + 1
        max_d = max(d for d in distances if d < float("inf"))
        distances[distances == float("inf")] = max_d + 1
        return distances

    @staticmethod
    def _normalize_adj(A):
        """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.where(D > 0, np.power(D, -0.5), 0)
        D_mat = np.diag(D_inv_sqrt)
        return D_mat @ A @ D_mat

    def build_spatial_only_adjacency(self):
        """For spatial-only ablation: same graph but no temporal edges will be used."""
        return self.build_adjacency()

    def get_edge_list(self):
        """Return edges as list of (src, dst) tuples for PyG compatibility."""
        edges = []
        # Player 1 intra-edges
        for (i, j) in COCO_SKELETON:
            edges.append((i, j))
            edges.append((j, i))

        if not self.single_player:
            # Player 2 intra-edges
            for (i, j) in COCO_SKELETON:
                edges.append((i + NUM_JOINTS, j + NUM_JOINTS))
                edges.append((j + NUM_JOINTS, i + NUM_JOINTS))
            # Inter-player edges
            if self.use_inter_player:
                for (i, j) in INTER_PLAYER_EDGES:
                    edges.append((i, j))
                    edges.append((j, i))

        return edges
