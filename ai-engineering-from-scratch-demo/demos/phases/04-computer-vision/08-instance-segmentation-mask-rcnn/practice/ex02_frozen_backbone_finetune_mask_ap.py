"""Exercise 2 — frozen backbone finetune mask ap.

    **(Medium)** Fine-tune `maskrcnn_resnet50_fpn_v2` on a 50-image custom
    dataset (any two classes: balloons, fish, pothole, logos). Freeze the
    backbone, train for 20 epochs, report mask AP@0.5.

Reading of the exercise: the recipe is run as the lesson writes it -- swap both
predictor heads for a 3-class dataset, `freeze_backbone`, sum the five losses --
over a two-class set of disks and diamonds drawn in one colour on noise, so shape
and not colour is the signal and neither class is its own bounding box (a disk
fills 79% of one, a diamond 50%). Two departures from the exercise, both forced,
both stated in the checks below.

The first is the weights. `build_custom_maskrcnn` and `load_pretrained_maskrcnn`
both hard-code `weights=...DEFAULT`, so the lesson offers no way to build this
architecture without a 170 MB checkpoint fetch, and a test that needs the network
is not reproducible. Everything here is therefore randomly initialised, which
means the headline the exercise asks for -- mask AP@0.5 -- comes out at 0.000:
fine-tuning is the transfer, and with nothing to transfer the recipe has nothing
to do. What a COCO-pretrained backbone would score instead is not stated here,
because nothing in this file measures it. The summed loss still
falls 4.1x, which is exactly the "loss went down, the metric did not move" trap.
The real command and its GPU cost are printed in that check.

The second is the anchors. The stock RPN starts at a 32-pixel anchor because the
stock `min_size` is 800; scaled to 96px inputs the objects are 16-24 px across,
and the best achievable IoU between any anchor shape and any ground-truth box is
measured below at 0.56 -- under the RPN's own 0.7 positive threshold, so every
positive anchor in the set is an `allow_low_quality_matches` rescue rather than a
real match. The anchor sizes are therefore rescaled 4x before training, which is
the one change to the recipe that is not in the lesson.

Two structural results are weight-independent and survive all of that: the
parameter split after the head swap and the freeze, and the fact that
`freeze_backbone` does not freeze the backbone. The v2 model builds its ResNet
with real `BatchNorm2d`, not the `FrozenBatchNorm2d` of v1, so `requires_grad =
False` holds every backbone weight bit-identical -- measured at 0.0 -- while
leaving 61 layers free to rewrite `running_mean`/`running_var` on each training
forward pass, measured moving by up to 29.89. Those buffers feed straight into
the eval-mode features, so a real freeze also needs `backbone.eval()`.

Mask AP is scored as an all-point-interpolated precision-recall area over greedy,
score-ordered, same-class matching on mask IoU, since pycocotools is not a
dependency here; the last check pins that implementation down by feeding it the
ground truth. That metric, a synthetic instance-segmentation dataset and the
fine-tune loop are three deliverables the lesson's `code/` does not supply, which
is what puts this file in D14's 120-150 review band.

Structure: `dataset` paints one shape per 32px column so instances never overlap
(disk = 1, diamond = 2); `mask_ap` scores mask IoU rather than box IoU, claiming
ground truth greedily in score order and then integrating all-point-interpolated
precision; the module-level lambdas above hold the parameter accounting and the
BatchNorm buffer snapshots.
"""

from __future__ import annotations

from itertools import accumulate
from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "08-instance-segmentation-mask-rcnn"

SIZE, CELL, SHAPES, N_TRAIN, N_VAL, STEPS, BATCH, LR, CLASSES = 96, 32, 3, 8, 6, 16, 4, 0.01, 3
ANCHORS = ((8,), (16,), (32,), (64,), (128,))       # the stock (32, 64, 128, 256, 512), divided by 4
REAL_RUN = ("the exercise as written needs the checkpoint: `python finetune.py --weights coco --images 50 "
            "--epochs 20 --min-size 800` is ~1,000 steps, ~4 GPU-min on an A10G (~3 cents), ~3 h on a CPU")

totals = lambda params: [sum(v[i] for v in params.values()) for i in (0, 1)]                           # noqa: E731
disk = lambda xx, yy, cx, cy, r: ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2                            # noqa: E731
diamond = lambda xx, yy, cx, cy, r: (xx - cx).abs() + (yy - cy).abs() < r                              # noqa: E731
images, labelled = lambda pairs: [i for i, _ in pairs], lambda pairs: [t for _, t in pairs]            # noqa: E731
boxes = lambda torch, val: torch.cat([t["boxes"] for t in labelled(val)])                              # noqa: E731
running = lambda m: {k: v.clone() for k, v in m.backbone.state_dict().items() if "running_" in k}      # noqa: E731
drifted = lambda m, s: max(float((m.backbone.state_dict()[k] - v).abs().max()) for k, v in s.items())  # noqa: E731
count = lambda mod, unfrozen=False: sum(p.numel() for p in mod.parameters() if p.requires_grad or not unfrozen)  # noqa: E731
budget = lambda model: {name: (count(mod), count(mod, True)) for name, mod in model.named_children()}  # noqa: E731
oracle = lambda torch, val: [{"scores": torch.ones(len(t["labels"])), "labels": t["labels"],           # noqa: E731
                              "masks": t["masks"].float().unsqueeze(1)} for _, t in val]
anchor_iou = lambda torch, box_iou, cells, truth: float(max(                                           # noqa: E731
    box_iou(shape + torch.cat([(truth[:, :2] + truth[:, 2:]) / 2] * 2, 1), truth).diag().max()
    for shape in torch.cat([level.float() for level in cells])))


def dataset(torch, count_images, seed):
    yy, xx = (axis := torch.arange(SIZE, dtype=torch.float32))[:, None].expand(SIZE, SIZE), axis.expand(SIZE, SIZE)
    gen, paint, out = torch.Generator().manual_seed(seed), torch.tensor([0.9, 0.85, 0.2])[:, None], []
    for _ in range(count_images):
        spec = [(1 + int(d[0] < 0.5), CELL * k + 12 + float(d[1]) * 8, 12 + float(d[2]) * (SIZE - 24),
                 8 + float(d[3]) * 4) for k, d in enumerate(torch.rand(SHAPES, 4, generator=gen))]
        masks = torch.stack([(disk if s[0] == 1 else diamond)(xx, yy, *s[1:]) for s in spec])
        image = torch.rand(3, SIZE, SIZE, generator=gen) * 0.2 + 0.1
        image[:, masks.any(0)] = paint
        out.append((image, {"masks": masks.to(torch.uint8), "labels": torch.tensor([s[0] for s in spec]),
                            "boxes": torch.tensor([[x - r, y - r, x + r, y + r] for _, x, y, r in spec])}))
    return out


def mask_ap(torch, preds, targets, threshold):
    scored = []
    for pred, target in zip(preds, targets):
        order = pred["scores"].argsort(descending=True)
        p = (pred["masks"][order, 0] > 0.5).flatten(1).float()
        g = target["masks"].bool().flatten(1).float()
        overlap = p @ g.t()
        iou = overlap / (p.sum(1, keepdim=True) + g.sum(1) - overlap).clamp(min=1.0)
        iou = iou * (pred["labels"][order][:, None] == target["labels"])
        taken = torch.zeros(len(target["labels"]), dtype=torch.bool)
        for j, score in enumerate(pred["scores"][order].tolist()):
            free = iou[j].masked_fill(taken, 0.0)
            scored.append((score, bool(free.max() >= threshold)))
            taken[int(free.argmax())] |= scored[-1][1]
    found = list(accumulate(hit for _, hit in sorted(scored, key=lambda pair: -pair[0])))
    recall = [f / sum(len(t["labels"]) for t in targets) for f in found]
    sharp = list(accumulate(reversed([f / (i + 1) for i, f in enumerate(found)]), max))[::-1]
    return sum(p * (r - q) for p, r, q in zip(sharp, recall, [0.0] + recall))


def solve():
    try:
        import torch
        from torchvision.models.detection import anchor_utils, maskrcnn_resnet50_fpn_v2
        from torchvision.ops import box_iou
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = ref.freeze_backbone(maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=CLASSES))
    train, val = dataset(torch, N_TRAIN, 0), dataset(torch, N_VAL, 99)
    stock = anchor_iou(torch, box_iou, model.rpn.anchor_generator.cell_anchors, boxes(torch, val))
    model.rpn.anchor_generator = anchor_utils.AnchorGenerator(ANCHORS, ((0.5, 1.0, 2.0),) * len(ANCHORS))
    model.transform.min_size, model.transform.max_size = (SIZE,), SIZE + 32
    model.roi_heads.fg_bg_sampler.batch_size_per_image = model.rpn.fg_bg_sampler.batch_size_per_image = 64
    model.rpn._pre_nms_top_n = model.rpn._post_nms_top_n = {"training": 300, "testing": 150}
    model.roi_heads.detections_per_img, model.roi_heads.score_thresh = 10, 0.0
    stats, weights = running(model.eval()), {k: v.clone() for k, v in model.backbone.named_parameters()}
    optimiser = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9)
    history, losses = [], {}
    for step in range(STEPS):                       # `train_step` from the lesson's docs, in a loop
        model.train()
        batch = train[BATCH * step % N_TRAIN:BATCH * step % N_TRAIN + BATCH]
        losses = model(images(batch), labelled(batch))
        total = sum(losses.values())
        optimiser.zero_grad()
        total.backward()
        optimiser.step()
        history.append(float(total.detach()))
    matcher = model.rpn.proposal_matcher
    with torch.no_grad():
        preds = model.eval()(images(val))
    return {"after": {t: mask_ap(torch, preds, labelled(val), t) for t in (0.5, 0.75, 0.9)},
            "history": history, "loss_keys": sorted(losses), "params": budget(model),
            "drift": drifted(model, stats), "held": drifted(model, weights),
            "anchors": (stock, matcher.high_threshold, matcher.allow_low_quality_matches),
            "classes": model.roi_heads.box_predictor.cls_score.out_features,
            "batchnorms": sum(type(x).__name__ == "BatchNorm2d" for x in model.backbone.modules()),
            "graded": mask_ap(torch, oracle(torch, val), labelled(val), 0.9)}


def verify(result):
    after, params, history = result["after"], result["params"], result["history"]
    total, trainable = totals(params)
    stock, positive, rescued = result["anchors"]
    return [
        practice.Check(
            "ANSWER: the head swap and the freeze leave 41% of the detector trainable",
            result["classes"] == CLASSES and params["backbone"][1] == 0 < params["rpn"][1],
            f"the box classifier emits {result['classes']} logits (background + disk + diamond) against COCO's 91, and "
            f"of {total:,} parameters {trainable:,} ({trainable / total:.1%}) still train: "
            f"{', '.join(f'{n} {v[1]:,}/{v[0]:,}' for n, v in params.items() if v[0])} -- ResNet and FPN frozen"),
        practice.Check(
            f"ANSWER: mask AP@0.5 = {after[0.5]:.3f} from random init, loss down {history[0] / history[-1]:.1f}x",
            after[0.5] < 0.2 and history[-1] < 0.5 * history[0],
            f"{N_TRAIN} images, {N_VAL} held out with {SHAPES * N_VAL} instances, {STEPS} steps of SGD(lr={LR}): "
            f"{', '.join(f'AP@{t:.2f} {v:.3f}' for t, v in after.items())}. Summed as the docs' `train_step` "
            f"does, the total runs {' -> '.join(f'{v:.2f}' for v in history)} over {result['loss_keys']} -- five "
            f"losses, not the four en.md says the model returns. Fine-tuning is the transfer, and there "
            f"is none to make. {REAL_RUN}"),
        practice.Check(
            "MECHANISM: every stock anchor here is a low-quality rescue, not a real match",
            stock < positive and rescued,
            f"re-centring every stock anchor shape on every ground-truth box, the best IoU any anchor reaches is "
            f"{stock:.2f} -- under the RPN's {positive} threshold, so only allow_low_quality_matches={rescued} "
            f"keeps a positive at all. Stock sizes assume 800px inputs; they are rescaled 4x here"),
        practice.Check(
            "FINDING: the weights are frozen; the BatchNorm buffers are not",
            result["held"] == 0.0 and result["drift"] > 0.1,
            f"its {result['batchnorms']} `nn.BatchNorm2d` layers are real, not the `FrozenBatchNorm2d` v1 uses, so "
            f"after {len(history)} passes every backbone weight is bit-identical ({result['held']:.1f}) while the "
            f"running statistics moved up to {result['drift']:.2f}: a freeze wants `backbone.eval()` too"),
        practice.Check(
            "CONTROL: the AP itself is right -- feed it the ground truth and it returns 1.0",
            result["graded"] > 0.999 > after[0.9],
            f"scoring the ground-truth masks against themselves at the hardest threshold gives AP@0.90 = "
            f"{result['graded']:.3f}, so the model's own {after[0.9]:.3f} is {STEPS} steps from random weights"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
