"""Train a small text-detection network and export it to ONNX.

A segmentation net, not a detector: it predicts, for every pixel, whether that
pixel is part of a line of print. Boxes come from connected components of the
probability map afterwards. That is how DB and similar text detectors work, and
it suits this problem because industrial labels carry text at any orientation,
which box-regression detectors handle poorly.

Deliberately small -- around 100k parameters, predicting at quarter resolution.
It has one job on one kind of imagery, it has to train on a CPU in minutes, and
it has to run on a line PC inside an inspection cycle. Accuracy on ImageNet is
not the goal; separating print from barcodes and brushed metal is.

    python textnet_train.py --minutes 12
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

import textnet_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "models")


def block(i, o, stride=1):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, stride, 1, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


class TextNet(nn.Module):
    """Encoder-decoder with skips. Output is quarter resolution."""

    def __init__(self, w=16):
        super().__init__()
        self.e1 = nn.Sequential(block(1, w), block(w, w))            # 1/1
        self.e2 = nn.Sequential(block(w, w * 2, 2), block(w * 2, w * 2))   # 1/2
        self.e3 = nn.Sequential(block(w * 2, w * 4, 2), block(w * 4, w * 4))  # 1/4
        self.e4 = nn.Sequential(block(w * 4, w * 4, 2), block(w * 4, w * 4))  # 1/8
        self.up1 = nn.ConvTranspose2d(w * 4, w * 4, 2, 2)            # ->1/4
        self.d1 = nn.Sequential(block(w * 8, w * 4), block(w * 4, w * 2))
        self.head = nn.Conv2d(w * 2, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d = self.d1(torch.cat([self.up1(e4), e3], 1))
        return self.head(d)                    # logits at 1/4 resolution


def dice_bce(logits, target):
    """Dice alongside BCE: text is a small fraction of the image, and BCE alone
    is happy to predict 'no text everywhere' and call it a good score."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return bce + (1 - num / den).mean()


def downsample_mask(y):
    return torch.nn.functional.max_pool2d(y, 4)


def evaluate(model, rng, n=24):
    model.eval()
    xs, ys = D.batch(n, rng)
    with torch.no_grad():
        logits = model(torch.from_numpy(xs))
        t = downsample_mask(torch.from_numpy(ys))
        p = (torch.sigmoid(logits) > 0.5).float()
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
    model.train()
    prec = tp / max(1e-6, tp + fp)
    rec = tp / max(1e-6, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--resume", type=int, default=1)
    a = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count()))
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    val_rng = np.random.default_rng(999)

    model = TextNet(a.width)
    n_params = sum(p.numel() for p in model.parameters())
    print("TextNet: %d parameters" % n_params)

    # Resume if a checkpoint is there. Training runs in short chunks here, so
    # every chunk has to continue the last one rather than start over.
    ckpt = os.path.join(OUT, "textnet.pt")
    if a.resume and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt))
        print("resumed from", ckpt)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    deadline = time.time() + a.minutes * 60
    sched = None

    step, t0, losses = 0, time.time(), []
    best = 0.0
    if a.resume and os.path.exists(ckpt):
        best = evaluate(model, np.random.default_rng(999))[2]
        print("starting F1 %.3f" % best)
    os.makedirs(OUT, exist_ok=True)
    while time.time() < deadline:
        xs, ys = D.batch(a.batch, rng)
        x = torch.from_numpy(xs)
        y = downsample_mask(torch.from_numpy(ys))
        logits = model(x)
        loss = dice_bce(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        step += 1
        if step % 25 == 0:
            prec, rec, f1 = evaluate(model, val_rng)
            left = (deadline - time.time()) / 60.0
            print("  step %4d  loss %.4f  P %.2f R %.2f F1 %.2f  %.1f min left"
                  % (step, np.mean(losses[-25:]), prec, rec, f1, left), flush=True)
            if f1 > best:
                best = f1
                torch.save(model.state_dict(), os.path.join(OUT, "textnet.pt"))

    print("trained %d steps in %.1f min, best F1 %.3f"
          % (step, (time.time() - t0) / 60.0, best))

    model.load_state_dict(torch.load(os.path.join(OUT, "textnet.pt")))
    model.eval()
    dummy = torch.zeros(1, 1, D.SIZE, D.SIZE)
    path = os.path.join(OUT, "textnet.onnx")
    torch.onnx.export(
        model, dummy, path, input_names=["image"], output_names=["textmap"],
        dynamic_axes={"image": {2: "h", 3: "w"}, "textmap": {2: "h", 3: "w"}},
        opset_version=18)   # native; asking for 12 forced a down-conversion that printed an alarming but harmless RuntimeError
    print("exported", path, "%.1f kB" % (os.path.getsize(path) / 1024))


if __name__ == "__main__":
    main()
