import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from dataset import STGCNDataset
from models.stgcn import STGCN, build_hand_edge_index


def set_seed(seed: int) -> None:
    # Make experiments more reproducible.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    # soft_targets: (N, num_classes)
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def maybe_label_smooth_one_hot(one_hot: torch.Tensor, epsilon: float) -> torch.Tensor:
    if epsilon <= 0:
        return one_hot
    num_classes = one_hot.size(1)
    return one_hot * (1.0 - epsilon) + (epsilon / num_classes)


def detect_velocity_channels(num_channels: int) -> bool:
    # Heuristic: if C in {4, 6}, assume it's [pos, vel] concatenation.
    return num_channels in (4, 6)


def augment_sequence(
    x: torch.Tensor,
    jitter_std: float,
    time_warp: float,
    flip_prob: float,
    drop_frames: int,
) -> torch.Tensor:
    # x: (T, V, C) on CPU
    t, v, c = x.shape
    has_vel = detect_velocity_channels(c)
    pos_c = c // 2 if has_vel else c

    # 1) Jitter: small Gaussian noise.
    if jitter_std > 0:
        noise = torch.randn_like(x[:, :, :pos_c]) * jitter_std
        x[:, :, :pos_c] = x[:, :, :pos_c] + noise

    # 2) Flip: mirror X (and velocity X) with some probability.
    if flip_prob > 0 and torch.rand(()) < flip_prob:
        # Position x is channel 0. Velocity x is channel pos_c if present.
        x[:, :, 0] = -x[:, :, 0]
        if has_vel:
            x[:, :, pos_c + 0] = -x[:, :, pos_c + 0]

    # 3) Time-warp: resample sequence length by a random scale then crop/pad back.
    if time_warp > 0:
        # scale in [1-time_warp, 1+time_warp]
        scale = float((1 - time_warp) + (2 * time_warp) * torch.rand(()).item())
        new_len = max(2, int(round(t * scale)))

        # Interpolate along time axis. Shape to (1, C, V, T) to use interpolate.
        xt = x.permute(2, 1, 0).unsqueeze(0)  # (1, C, V, T)
        xt = F.interpolate(xt, size=(v, new_len), mode="bilinear", align_corners=False)
        x = xt.squeeze(0).permute(2, 1, 0).contiguous()  # (new_len, V, C)

        # Crop/pad to original length t.
        if x.size(0) > t:
            x = x[:t]
        elif x.size(0) < t:
            pad = x[-1:].repeat(t - x.size(0), 1, 1)
            x = torch.cat([x, pad], dim=0)

    # 4) Random drop frames: replace random frames with previous frame.
    if drop_frames > 0 and t > 2:
        k = min(drop_frames, t - 1)
        idx = torch.randperm(t - 1)[:k] + 1
        x[idx] = x[idx - 1]

    return x


def make_collate_fn(
    jitter_std: float,
    time_warp: float,
    flip_prob: float,
    drop_frames: int,
):
    # Apply augmentation on CPU before moving batch to GPU.
    def collate(batch):
        xs, ys = zip(*batch)
        xs = [augment_sequence(x.clone(), jitter_std, time_warp, flip_prob, drop_frames) for x in xs]
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

    return collate


def split_dataset(dataset: STGCNDataset, val_ratio: float) -> Tuple[STGCNDataset, STGCNDataset]:
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


def save_label_map(label_map: Dict[str, int], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "labels.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        x = x.permute(0, 3, 1, 2)  # (N, C, T, V)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += x.size(0)

    return total_loss / total_count, total_correct / total_count


def update_confusion_matrix(cm: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
    for t, p in zip(targets, preds):
        cm[t, p] += 1


def eval_epoch(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.permute(0, 3, 1, 2)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).sum().item()
            total_count += x.size(0)
            update_confusion_matrix(confusion, preds.cpu(), y.cpu())

    return total_loss / total_count, total_correct / total_count, confusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to .npz file")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine", "step"], default="none")
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disable)")
    parser.add_argument("--seed", type=int, default=42)

    # Augmentation
    parser.add_argument("--aug-jitter-std", type=float, default=0.0)
    parser.add_argument("--aug-time-warp", type=float, default=0.0, help="Warp strength in [0,1]")
    parser.add_argument("--aug-flip-prob", type=float, default=0.0)
    parser.add_argument("--aug-drop-frames", type=int, default=0)

    # Mixup
    parser.add_argument("--mixup-alpha", type=float, default=0.0)

    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = STGCNDataset(args.data)
    train_set, val_set = split_dataset(dataset, args.val_ratio)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = build_hand_edge_index()
    model = STGCN(
        in_channels=dataset.sequences.shape[-1],
        num_classes=len(dataset.label_to_index),
        edge_index=edge_index,
        dropout=args.dropout,
    )
    model = model.to(device)

    collate_fn = make_collate_fn(
        jitter_std=args.aug_jitter_std,
        time_warp=args.aug_time_warp,
        flip_prob=args.aug_flip_prob,
        drop_frames=args.aug_drop_frames,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    best_acc = 0.0
    epochs_no_improve = 0
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train step with optional mixup.
        if args.mixup_alpha > 0:
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            num_classes = len(dataset.label_to_index)

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                x = x.permute(0, 3, 1, 2)  # (N, C, T, V)

                # Sample lambda from Beta(alpha, alpha)
                lam = torch.distributions.Beta(args.mixup_alpha, args.mixup_alpha).sample(()).to(device)
                perm = torch.randperm(x.size(0), device=device)
                x2 = x[perm]
                y2 = y[perm]
                x_mix = lam * x + (1 - lam) * x2

                y1_oh = F.one_hot(y, num_classes=num_classes).float()
                y2_oh = F.one_hot(y2, num_classes=num_classes).float()
                y1_oh = maybe_label_smooth_one_hot(y1_oh, args.label_smoothing)
                y2_oh = maybe_label_smooth_one_hot(y2_oh, args.label_smoothing)
                y_mix = lam * y1_oh + (1 - lam) * y2_oh

                optimizer.zero_grad()
                logits = model(x_mix)
                loss = soft_cross_entropy(logits, y_mix)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_count += x.size(0)

            train_loss = total_loss / total_count
            train_acc = total_correct / total_count
        else:
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, val_acc, confusion = eval_epoch(
            model,
            val_loader,
            criterion,
            device,
            num_classes=len(dataset.label_to_index),
        )

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"lr {lr:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "stgcn_best.pt")
            save_label_map(dataset.label_to_index, output_dir)
            # Save confusion matrix for the best validation accuracy.
            torch.save(confusion, output_dir / "confusion_matrix.pt")
        else:
            if args.patience > 0:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch} (best acc {best_acc:.3f})")
                    break

    torch.save(model.state_dict(), output_dir / "stgcn_last.pt")


if __name__ == "__main__":
    main()
