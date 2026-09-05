"""Exercise 4 — early stopping with patience 20, over 1000 epochs.

    Implement early stopping: track test loss each epoch, save the best weights,
    and stop if test loss hasn't improved for 20 epochs. Run the regularized
    network for 1000 epochs. Report which epoch had the best test accuracy and how
    many epochs of computation you saved.

Reading of the exercise: both of its questions are answered literally in checks 1-2, and both
turn out to be less informative than they look — so checks 3-5 ask what the rule actually
selected, whether there was anything to stop, and what the number it is stopping on has already
been spent on. "The regularized network" is the lesson's own dropout + weight-decay
configuration, run beside an unregularized twin so the two answers can be compared.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "07-regularization"
HIDDEN, LR, EPOCHS, PATIENCE = 16, 0.05, 1000, 20
ARMS = {"plain": {}, "regularized": {"dropout_p": 0.3, "weight_decay": 0.001}}


def history(ref, data, **kwargs) -> tuple:
    """The lesson's own network and loop for the full 1000 epochs, and the trained net."""
    net = ref.RegularizedNetwork(HIDDEN, LR, **kwargs)
    with parity.quiet():
        rows = net.train_model(data[:150], data[150:], epochs=EPOCHS)
    return rows, net


def watch(test_loss) -> dict:
    """Early stopping read off the recorded curve: best-so-far, patience, restore point."""
    best, kept, wait, stop = float("inf"), 0, 0, None
    for epoch, value in enumerate(test_loss):
        if value < best:
            best, kept, wait = value, epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE and stop is None:
                stop = epoch + 1
                break
    return {"stop": stop, "kept": kept, "kept_loss": best,
            "saved": EPOCHS - (stop if stop is not None else EPOCHS)}


def summarise(rows) -> dict:
    """Everything the exercise asks for, plus what it does not."""
    train_loss = [r[0] for r in rows]
    test_loss, test_acc = [r[2] for r in rows], [r[3] for r in rows]
    top = max(test_acc)
    return {**watch(test_loss), "best_acc_epoch": test_acc.index(top), "top": top,
            "ties": test_acc.count(top), "floor": min(test_loss),
            "floor_epoch": test_loss.index(min(test_loss)),
            "rises": sum(test_loss[i] > test_loss[i - 1] for i in range(1, EPOCHS)),
            "train": (train_loss[0], train_loss[-1]), "last": test_loss[-1]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data, fresh = ref.make_circle_data(200, 42), ref.make_circle_data(200, 7)
    arms, nets = {}, {}
    for name, kwargs in ARMS.items():
        rows, net = history(ref, data, **kwargs)
        arms[name], nets[name] = summarise(rows), net
    with parity.quiet():
        holdout = {n: ref.RegularizedNetwork.evaluate(net, fresh) for n, net in nets.items()}
    return {"arms": arms, "holdout": holdout, "n": len(data[150:])}


def verify(result):
    arms, hold = result["arms"], result["holdout"]
    plain, reg = arms["plain"], arms["regularized"]
    return [
        practice.Check("ANSWER: the best test-accuracy epoch is the first of a 992-way tie",
                       plain["ties"] > 0.95 * EPOCHS and reg["ties"] > 0.95 * EPOCHS,
                       f"best test accuracy is {plain['top']:.1f}% first reached at epoch "
                       f"{plain['best_acc_epoch']} unregularized and {reg['top']:.1f}% at epoch "
                       f"{reg['best_acc_epoch']} regularized — but {plain['ties']} and "
                       f"{reg['ties']} of {EPOCHS} epochs *tie* at that accuracy. On "
                       f"{result['n']} held-out points at 100% there is no best epoch to report, "
                       f"only the first of a tie"),
        practice.Check("ANSWER: 889 epochs saved on the regularized network, 0 on the plain one",
                       reg["saved"] > 800 and plain["saved"] == 0,
                       f"patience {PATIENCE} stops the regularized arm at epoch {reg['stop']}, "
                       f"saving {reg['saved']} of {EPOCHS}. On the unregularized arm it never "
                       f"fires: its test loss is still setting records at epoch {EPOCHS - 1} "
                       f"({plain['floor']:.4f}, the best of the run), so it saves "
                       f"{plain['saved']} epochs"),
        practice.Check("FINDING: the rule keeps a worse model than the one it throws away",
                       reg["kept_loss"] > reg["floor"] and reg["floor_epoch"] > reg["stop"],
                       f"it restores epoch {reg['kept']} at test loss {reg['kept_loss']:.4f}, "
                       f"while the run's best test loss is {reg['floor']:.4f} at epoch "
                       f"{reg['floor_epoch']} — {reg['floor_epoch'] - reg['stop']} epochs after "
                       f"the stop, and still {reg['last']:.4f} at the end. Patience "
                       f"{PATIENCE} fired on a plateau, not on overfitting"),
        practice.Check("MECHANISM: there is no overfitting here to stop",
                       plain["floor_epoch"] == EPOCHS - 1 and reg["rises"] < 0.5 * EPOCHS,
                       f"train loss falls {plain['train'][0]:.4f} -> {plain['train'][1]:.4f} "
                       f"unregularized and {reg['train'][0]:.4f} -> {reg['train'][1]:.4f} "
                       f"regularized, and test loss falls with it — it rises on "
                       f"{plain['rises']} and {reg['rises']} of {EPOCHS - 1} epoch-to-epoch steps "
                       f"and never turns. 150 training points and {HIDDEN} hidden units is not a "
                       f"setting where the curve separates"),
        practice.Check("CONTROL: the number being stopped on is the number being reported",
                       all(acc < plain["top"] for _loss, acc in hold.values()),
                       f"the same {result['n']} points choose the stopping epoch, select the "
                       f"weights and score the result — and the {plain['top']:.1f}% they report "
                       f"does not survive a fresh 200-point draw, where the finished nets score "
                       + ", ".join(f"{n} {acc:.1f}% (loss {loss:.4f})"
                                   for n, (loss, acc) in hold.items())
                       + ". Nothing in the exercise separates the set it stops on from the set "
                       "it reports"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
