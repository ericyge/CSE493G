"""Many-to-one LSTM for tennis serve type classification (Flat / Kick / Slice)."""

from __future__ import annotations

import torch
import torch.nn as nn

# Matches Phase 1 tensors: (batch, seq_len=3, input_size=6)
SEQ_LEN = 3
INPUT_SIZE = 6
NUM_CLASSES = 3

CLASS_NAMES = ("Flat", "Kick", "Slice")


class ServeLSTM(nn.Module):
    """
    Many-to-one LSTM over fixed trajectory milestones: hit → net → bounce.

    Input:  (batch, 3, 6) — xyz (+ padded deltas at hit; deltas at net/bounce)
    Output: (batch, 3) logits for Flat / Kick / Slice

    Uses the final hidden state of the LSTM, then a linear classifier.
    Apply softmax outside the model (e.g. for inference) or use CrossEntropyLoss
    on logits during training.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        head_in = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) float tensor

        Returns:
            logits: (batch, num_classes)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (batch, seq_len, features), got shape {tuple(x.shape)}")
        if x.size(1) != SEQ_LEN or x.size(2) != self.input_size:
            raise ValueError(
                f"Expected (batch, {SEQ_LEN}, {self.input_size}), got {tuple(x.shape)}"
            )

        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            last_forward = h_n[-2]
            last_backward = h_n[-1]
            context = torch.cat([last_forward, last_backward], dim=1)
        else:
            context = h_n[-1]

        return self.head(context)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities (batch, num_classes)."""
        return torch.softmax(self.forward(x), dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted class indices (batch,)."""
        return self.forward(x).argmax(dim=1)


def build_model(
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    device: torch.device | str | None = None,
) -> ServeLSTM:
    """Construct model and optionally move to device."""
    model = ServeLSTM(
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )
    if device is not None:
        model = model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device=device)
    batch = torch.randn(8, SEQ_LEN, INPUT_SIZE, device=device)
    logits = model(batch)
    probs = model.predict_proba(batch)
    print(model)
    print(f"Input:  {tuple(batch.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Probs:  {tuple(probs.shape)} (sum per row ≈ 1)")
    print(f"Sample preds: {[CLASS_NAMES[i] for i in model.predict(batch[:3]).tolist()]}")
