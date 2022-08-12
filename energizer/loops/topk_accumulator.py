import torch
from torch import Tensor
from torchmetrics import Metric


class AccumulateTopK(Metric):
    full_state_update = False

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("topk_scores", torch.tensor([float("-inf")] * self.k))
        self.add_state("indices", torch.ones(self.k, dtype=torch.int64).neg())
        self.add_state("size", torch.tensor(0, dtype=torch.int64))

    def update(self, scores: Tensor) -> None:
        batch_size = scores.numel()

        # indices with respect to the pool dataset and scores
        current_indices = torch.arange(self.size, self.size + batch_size, 1)

        # compute topk comparing current batch with the states
        all_scores = torch.cat([self.topk_scores, scores], dim=0)
        all_indices = torch.cat([self.indices, current_indices], dim=0)

        # aggregation
        topk_scores, idx = torch.topk(all_scores, k=min(self.k, batch_size))

        self.topk_scores.copy_(topk_scores)
        self.indices.copy_(all_indices[idx])
        self.size += batch_size

        # print("current batch_size:", batch_size)
        # print("current_indices:", current_indices)
        # print("size:", self.size)
        # print("all_scores:", all_scores)
        # print("all_indices:", all_indices)
        # print("top_scores:", topk_scores)
        # print("top_indices:", all_indices[idx], "\n")

    def compute(self) -> Tensor:
        # print("compute indices:", self.indices)
        return self.indices
