"""Exercise 4 — a custom Dataset, and the two methods that are the whole contract.

    **Build a custom Dataset.** Download Fashion-MNIST (same format as MNIST but
    with clothing items). Implement a `FashionMNISTDataset(Dataset)` class with
    `__getitem__` and `__len__`. Train the same MLP and compare accuracy.
    Fashion-MNIST is harder -- expect ~88% vs ~98%.

Reading of the exercise: the download is the one part that cannot run here, so the class is
built against the lesson's own IDX loader signature and fed a seeded 784-D stand-in whose
class separation is tuned to reproduce the 88/98 gap the exercise predicts — check 2 carries
the real command. The rest is checkable exactly: check 1 holds the class to byte-parity with
`TensorDataset` through the lesson's own `create_loaders`, and checks 3-4 break the contract.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import torch
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "03-deep-learning-core", "11-intro-to-pytorch"
N_TRAIN, N_TEST, BATCH, EPOCHS, SEED = 2048, 1000, 64, 5, 7
EASY, HARD = 0.18, 0.12          # class separations standing in for MNIST and Fashion-MNIST
SOURCE = "https://storage.googleapis.com/cvdf-datasets/fashion-mnist/ (4 files, ~30 MB)"


class FashionMNISTDataset(torch.utils.data.Dataset):
    """The exercise's class: two methods, and `length` so check 3 can lie about one."""

    def __init__(self, images, labels, length=None, numpy_rows=False):
        self.images, self.labels = images, labels
        self.length = len(labels) if length is None else length
        self.numpy_rows = numpy_rows

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.numpy_rows:
            return self.images[index].numpy().astype("float64"), int(self.labels[index])
        return self.images[index], self.labels[index]


def blobs(separation, seed=0) -> tuple:
    """Seeded stand-in with MNIST's shapes — 784 features, 10 classes, 0..1 pixels."""
    gen = torch.Generator().manual_seed(seed)
    mid = torch.randn(10, 784, generator=gen) * separation
    ys = [torch.arange(n) % 10 for n in (N_TRAIN, N_TEST)]
    xs = [mid[y] + torch.randn(len(y), 784, generator=gen) for y in ys]
    return xs[0], ys[0], xs[1], ys[1]


def batches(dataset, shuffle, size=BATCH) -> list:
    loader = torch.utils.data.DataLoader(dataset, batch_size=size, shuffle=shuffle,
                                         generator=torch.Generator().manual_seed(SEED))
    return [(x.clone(), y.clone()) for x, y in loader]


def same(left, right) -> float:
    return max(max(float((a[0] - b[0]).abs().max()), float((a[1] - b[1]).abs().max()))
               for a, b in zip(left, right))


def accuracy(ref, separation) -> float:
    """The lesson's own model, loop and evaluate, over the custom Dataset."""
    xs, ys, xt, yt = blobs(separation)
    train = torch.utils.data.DataLoader(FashionMNISTDataset(xs, ys), batch_size=BATCH,
                                        shuffle=True, generator=torch.Generator().manual_seed(SEED))
    test = torch.utils.data.DataLoader(FashionMNISTDataset(xt, yt), batch_size=256)
    torch.manual_seed(0)
    model, crit, dev = ref.MNISTModel(), torch.nn.CrossEntropyLoss(), torch.device("cpu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _epoch in range(EPOCHS):
        ref.train_one_epoch(model, train, crit, opt, dev)
    return ref.evaluate(model, test, crit, dev)[1]


def lying_len(xs, ys, length) -> str:
    """What the DataLoader does when `__len__` does not match `__getitem__`'s range."""
    try:
        return f"{sum(len(y) for _x, y in batches(FashionMNISTDataset(xs, ys, length), False))} rows"
    except IndexError as exc:
        return " ".join(str(exc).split())[:64]


def wrong_dtype(ref, xs, ys) -> tuple:
    x, _y = batches(FashionMNISTDataset(xs, ys, numpy_rows=True), False)[0]
    torch.manual_seed(0)
    try:
        ref.MNISTModel()(x)
    except RuntimeError as exc:
        return str(x.dtype), " ".join(str(exc).split())[:72]
    return str(x.dtype), "no error"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pytorch_intro")
    xs, ys, xt, yt = blobs(EASY)
    theirs = ref.create_loaders(xs, ys, xt, yt, batch_size=BATCH)[1]
    return {"ordered": same(batches(FashionMNISTDataset(xs, ys), False),
                            batches(torch.utils.data.TensorDataset(xs, ys), False)),
            "shuffled": same(batches(FashionMNISTDataset(xs, ys), True),
                             batches(torch.utils.data.TensorDataset(xs, ys), True)),
            "theirs": same(batches(FashionMNISTDataset(xt, yt), False, 256), list(theirs)),
            "easy": accuracy(ref, EASY), "hard": accuracy(ref, HARD),
            "over": lying_len(xs, ys, len(ys) + 1), "under": lying_len(xs, ys, len(ys) - 4),
            "dtype": wrong_dtype(ref, xs, ys), "n": len(ys)}


def verify(result):
    dtype, message = result["dtype"]
    return [
        practice.Check("ANSWER: the custom Dataset is a drop-in for TensorDataset, batch for "
                       "batch",
                       max(result["ordered"], result["shuffled"], result["theirs"]) == 0.0,
                       f"the same {result['n']} rows through a DataLoader of batch {BATCH}: "
                       f"{result['ordered']:.1f} difference unshuffled, {result['shuffled']:.1f} "
                       f"shuffled from the same generator seed, and {result['theirs']:.1f} against "
                       f"the loader the lesson's own `create_loaders` builds. `__len__` and "
                       f"`__getitem__` are the entire interface"),
        practice.Check("ANSWER: the harder task reproduces the ~88 vs ~98 the exercise predicts",
                       result["easy"] > 0.97 and 0.84 < result["hard"] < 0.92,
                       f"the same MLP, optimizer and {EPOCHS} epochs over the custom Dataset: "
                       f"{100 * result['easy']:.1f}% at class separation {EASY} against "
                       f"{100 * result['hard']:.1f}% at {HARD}. The real run is this file with "
                       f"`blobs` replaced by the lesson's own `load_images`/`load_labels` over "
                       f"{SOURCE} — the only part of the exercise that needs the network"),
        practice.Check("MECHANISM: `__len__` decides how many indices are drawn, and nothing "
                       "checks it",
                       "out of bounds" in result["over"] and result["under"].startswith("2044"),
                       f"overstate it by one and the sampler asks for an index the data does not "
                       f"have: {result['over']}. Understate it by four and the loader yields "
                       f"{result['under']} of {result['n']} — silently, with no warning. The "
                       f"sampler trusts `__len__`; `__getitem__` is the only thing that can "
                       f"object, and only by raising"),
        practice.Check("FINDING: `__getitem__` may return the wrong dtype and the collate will "
                       "take it",
                       dtype == "torch.float64" and "dtype" in message,
                       f"returning each row as a numpy float64 array collates to a "
                       f"{dtype} batch — `default_collate` converts whatever it is handed. The "
                       f"model is the first thing to object: {message}. A Dataset that returns "
                       f"the wrong precision is a runtime error one layer away, not a load error"),
        practice.Check("CONTROL: shuffling is the sampler's, not the dataset's",
                       result["ordered"] == 0.0 and result["shuffled"] == 0.0,
                       "the same Dataset object gives identical batches to TensorDataset both "
                       "shuffled and not, because the permutation comes from the DataLoader's "
                       "RandomSampler and its generator. `__getitem__` never sees an epoch"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
