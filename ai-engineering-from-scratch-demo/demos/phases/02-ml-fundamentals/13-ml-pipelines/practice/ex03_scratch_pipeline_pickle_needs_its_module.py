"""Exercise 3 — sklearn round-trips bit for bit; the lesson's pipeline cannot load.

    Serialize your pipeline with `joblib.dump`. Load it in a separate script and
    run predictions. Verify the predictions are identical.

Reading of the exercise: "a separate script" is taken literally: a loader
script written to a temp directory and run in a fresh interpreter
(`sys.executable`), which loads the artifact, predicts on the lesson's 100-row
test split (`train_test_split_dict`) and prints the probabilities as JSON
(`repr` of a float round-trips exactly). The pipelines are exercise 1's sklearn
ColumnTransformer + logistic regression and the lesson's own `FullPipeline`,
both fit on the lesson's 400-row training split.

**ANSWER: identical.** All 100 probabilities from the separate process equal the
in-process ones exactly (max difference 0.0), and so do the 100 class labels.

**FINDING: the lesson's from-scratch pipeline dumps fine and then fails to
load.** A pickle stores a class as a module name plus a class name, not as
code. The fresh interpreter raises ModuleNotFoundError for the module that
defined `FullPipeline`; run as `python pipeline.py`, as the lesson is, that
module is `__main__`, which the loading script cannot supply either. The doc's
"serialized and deployed as one artifact" holds only if the code is shipped
with it.

**CONTROL: shipping the code fixes it.** The same loader, told to import the
lesson's `code/pipeline.py` under the recorded module name first, loads the
from-scratch pipeline and reproduces its 100 predictions exactly.

Structure: `LOADER` is the separate script; `load_elsewhere` runs it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from harness import parity, practice

PHASE, LESSON = "02-ml-fundamentals", "13-ml-pipelines"
NUM, CAT = ["age", "income", "score"], ["city", "plan"]
LOADER = """
import importlib.util, json, sys
import joblib
artifact, inputs, module, source = sys.argv[1:5]
if source:
    spec = importlib.util.spec_from_file_location(module, source)
    sys.modules[module] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[module])
try:
    pipe, X = joblib.load(artifact), joblib.load(inputs)
except Exception as exc:
    print(json.dumps({"error": type(exc).__name__}))
    raise SystemExit
out = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(X)
print(json.dumps({"out": [repr(float(v)) for v in out]}))
"""


def sk_pipeline():
    """Exercise 1's build, restated: exercise files do not import each other."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric, [0, 1, 2]), ("cat", categorical, [3, 4])])
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=1000))])


def load_elsewhere(tmp, name, pipe, X, source=""):
    """Dump `pipe` and `X`, load both in a fresh interpreter, return its JSON reply."""
    artifact, inputs = Path(tmp, f"{name}.joblib"), Path(tmp, f"{name}_X.joblib")
    joblib.dump(pipe, artifact)
    joblib.dump(X, inputs)
    script = Path(tmp, "loader.py")
    script.write_text(LOADER, encoding="utf-8")
    args = [str(artifact), str(inputs), type(pipe).__module__, str(source)]
    done = subprocess.run([sys.executable, str(script), *args], capture_output=True,
                          text=True, timeout=120, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "pipeline")
    train, test = ref.train_test_split_dict(ref.make_mixed_data(500))
    X_train, X_test = (np.array([d[c] for c in NUM + CAT], dtype=object).T for d in (train, test))
    sk = sk_pipeline().fit(X_train, train["target"])
    scratch = ref.FullPipeline(ref.LogisticRegressionSimple(0.05, 1000), NUM, CAT).fit(train)
    source = parity.lesson_dir(PHASE, LESSON) / "code" / "pipeline.py"
    with tempfile.TemporaryDirectory() as tmp:
        sk_out = load_elsewhere(tmp, "sk", sk, X_test)
        bare = load_elsewhere(tmp, "scratch", scratch, test)
        shipped = load_elsewhere(tmp, "scratch", scratch, test, source)
    here = sk.predict_proba(X_test)[:, 1]
    there = np.array([float(v) for v in sk_out["out"]])
    return {
        "n": len(here),
        "sk_gap": float(np.max(np.abs(here - there))),
        "labels_equal": bool(np.array_equal(here >= 0.5, there >= 0.5)),
        "bare": bare,
        "shipped_equal": [float(v) for v in shipped.get("out", [])]
        == scratch.predict(test).astype(float).tolist(),
        "module": type(scratch).__module__,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: predictions from the separate script are identical",
            result["sk_gap"] == 0.0 and result["labels_equal"] and result["n"] == 100,
            f"{result['n']} probabilities from a fresh interpreter differ from the in-process "
            f"ones by at most {result['sk_gap']}; the class labels match too",
        ),
        practice.Check(
            "FINDING: the lesson's from-scratch pipeline dumps but cannot load elsewhere",
            result["bare"] == {"error": "ModuleNotFoundError"},
            f"joblib.load in the separate script returns {result['bare']}: the pickle names "
            f"module {result['module']!r} and the class, not their code; run as a script that "
            "module would be __main__, equally absent from the loader",
        ),
        practice.Check(
            "CONTROL: shipping the lesson's code with the artifact fixes the load",
            result["shipped_equal"],
            "importing code/pipeline.py under the recorded module name first, the loader "
            f"reproduces all {result['n']} from-scratch predictions exactly",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
