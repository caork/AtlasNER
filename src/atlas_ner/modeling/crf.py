"""Conditional Random Field for sequence labeling."""

from __future__ import annotations

import torch
from torch import nn

from atlas_ner.data.schemes import parse_tag


def build_bioes_constraints(
    label_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build BIOES transition constraint masks.

    Returns:
        transition_mask: [num_labels, num_labels] bool — valid from→to transitions
        start_mask:      [num_labels] bool — valid start tags
        end_mask:        [num_labels] bool — valid end tags
    """
    n = len(label_names)
    transition_mask = torch.zeros(n, n, dtype=torch.bool)
    start_mask = torch.zeros(n, dtype=torch.bool)
    end_mask = torch.zeros(n, dtype=torch.bool)

    prefixes: list[str] = []
    types: list[str | None] = []
    for tag in label_names:
        if tag == "O":
            prefixes.append("O")
            types.append(None)
        else:
            p, t = parse_tag(tag)
            prefixes.append(p)
            types.append(t)

    for i in range(n):
        fp, ft = prefixes[i], types[i]
        # Start constraints: O, B-*, S-* can start a sequence
        if fp in ("O", "B", "S"):
            start_mask[i] = True
        # End constraints: O, E-*, S-* can end a sequence
        if fp in ("O", "E", "S"):
            end_mask[i] = True
        for j in range(n):
            tp, tt = prefixes[j], types[j]
            if fp in ("O", "E", "S"):
                # After O/E/S → O, B-*, S-*
                if tp in ("O", "B", "S"):
                    transition_mask[i, j] = True
            elif fp == "B":
                # After B-X → I-X, E-X only
                if tp in ("I", "E") and tt == ft:
                    transition_mask[i, j] = True
            elif fp == "I":
                # After I-X → I-X, E-X only
                if tp in ("I", "E") and tt == ft:
                    transition_mask[i, j] = True

    return transition_mask, start_mask, end_mask


_NEG_INF = -10000.0


class CRF(nn.Module):
    """Linear-chain CRF with optional BIOES constraints."""

    def __init__(
        self,
        num_labels: int,
        transition_mask: torch.Tensor | None = None,
        start_mask: torch.Tensor | None = None,
        end_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels) * 0.1)
        self.start_transitions = nn.Parameter(torch.randn(num_labels) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_labels) * 0.1)

        if transition_mask is None:
            transition_mask = torch.ones(num_labels, num_labels, dtype=torch.bool)
        if start_mask is None:
            start_mask = torch.ones(num_labels, dtype=torch.bool)
        if end_mask is None:
            end_mask = torch.ones(num_labels, dtype=torch.bool)
        self.register_buffer("transition_mask", transition_mask)
        self.register_buffer("start_mask", start_mask)
        self.register_buffer("end_mask", end_mask)

    def _constrained_transitions(self) -> torch.Tensor:
        return self.transitions.masked_fill(~self.transition_mask, _NEG_INF)

    def _constrained_start(self) -> torch.Tensor:
        return self.start_transitions.masked_fill(~self.start_mask, _NEG_INF)

    def _constrained_end(self) -> torch.Tensor:
        return self.end_transitions.masked_fill(~self.end_mask, _NEG_INF)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood CRF loss.

        Args:
            emissions: [batch, seq_len, num_labels]
            tags:      [batch, seq_len] (long)
            mask:      [batch, seq_len] (bool)

        Returns:
            Scalar loss (mean over batch).
        """
        forward_score = self._forward_algorithm(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        return (forward_score - gold_score).mean()

    def _forward_algorithm(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Log partition function via forward algorithm."""
        trans = self._constrained_transitions()  # [K, K]
        score = self._constrained_start() + emissions[:, 0]  # [B, K]

        for t in range(1, emissions.shape[1]):
            # score_prev[b, i] + trans[i, j] + emit[b, j]
            broadcast_score = score.unsqueeze(2)          # [B, K, 1]
            broadcast_emit = emissions[:, t].unsqueeze(1)  # [B, 1, K]
            next_score = broadcast_score + trans + broadcast_emit  # [B, K, K]
            next_score = torch.logsumexp(next_score, dim=1)       # [B, K]
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)

        score = score + self._constrained_end()
        return torch.logsumexp(score, dim=1)  # [B]

    def _score_sentence(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score the gold tag sequence."""
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score = score + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
            emit_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score = score + (trans_score + emit_score) * mask[:, t].float()

        # End transition at last valid position
        seq_lengths = mask.long().sum(dim=1) - 1  # [B]
        last_tags = tags.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        """Viterbi decoding."""
        batch_size, seq_len, _ = emissions.shape
        trans = self._constrained_transitions()

        score = self._constrained_start() + emissions[:, 0]  # [B, K]
        history: list[torch.Tensor] = []

        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)           # [B, K, 1]
            broadcast_emit = emissions[:, t].unsqueeze(1)  # [B, 1, K]
            next_score = broadcast_score + trans + broadcast_emit  # [B, K, K]
            next_score, indices = next_score.max(dim=1)    # [B, K]
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
            history.append(indices)

        score = score + self._constrained_end()

        best_tags_list: list[list[int]] = []
        _, best_last = score.max(dim=1)  # [B]
        for b in range(batch_size):
            length = int(mask[b].long().sum().item())
            best_path = [int(best_last[b].item())]
            for t in range(length - 2, -1, -1):
                best_path.append(int(history[t][b, best_path[-1]].item()))
            best_path.reverse()
            best_tags_list.append(best_path)
        return best_tags_list
