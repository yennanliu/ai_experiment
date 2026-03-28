from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from dvs.core import (
    DatasetManifest,
    ExperimentRecord,
    build_file_map,
    dataset_version_id,
    diff_manifests,
    experiment_id,
)
from dvs.store import Store


def cmd_init(store: Store) -> int:
    store.ensure_init()
    print(f"Initialized empty DVS store at {store.dvs_dir}")
    return 0


def cmd_add(store: Store, path: Path, tag: str | None) -> int:
    store.ensure_init()
    root = path.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1
    files = build_file_map(root)
    vid = dataset_version_id(str(root), files)
    manifest = DatasetManifest(version_id=vid, root=str(root), files=files, tag=tag)
    store.save_dataset(manifest)
    print(f"Dataset version {vid}  ({len(files)} files)")
    if tag:
        print(f"  tag: {tag}")
    return 0


def cmd_list_datasets(store: Store) -> int:
    for m in store.list_datasets():
        tag = f" [{m.tag}]" if m.tag else ""
        print(f"{m.version_id}{tag}  {len(m.files)} files  {m.created}")
        print(f"    root: {m.root}")
    return 0


def cmd_show_dataset(store: Store, version_id: str) -> int:
    m = store.load_dataset(version_id)
    print(json_pretty(m.to_json()))
    return 0


def cmd_diff(store: Store, a: str, b: str) -> int:
    ma = store.load_dataset(a)
    mb = store.load_dataset(b)
    added, removed, changed = diff_manifests(ma.files, mb.files)
    print(f"diff {a} .. {b}")
    print(f"  added ({len(added)}):")
    for p in added:
        print(f"    + {p}")
    print(f"  removed ({len(removed)}):")
    for p in removed:
        print(f"    - {p}")
    print(f"  changed ({len(changed)}):")
    for p in changed:
        print(f"    ~ {p}  ({ma.files[p][:8]}.. -> {mb.files[p][:8]}..)")
    return 0


def cmd_exp_create(
    store: Store, dataset_version: str, command: list[str], name: str | None, notes: str | None
) -> int:
    store.ensure_init()
    store.load_dataset(dataset_version)  # validate
    cwd = str(Path.cwd().resolve())
    eid = experiment_id(dataset_version, command, cwd)
    exp = ExperimentRecord(
        experiment_id=eid,
        dataset_version_id=dataset_version,
        command=command,
        workdir=cwd,
        name=name,
        notes=notes,
    )
    store.save_experiment(exp)
    print(f"experiment {eid}")
    return 0


def cmd_exp_list(store: Store) -> int:
    for e in store.list_experiments():
        label = e.name or "(no name)"
        print(f"{e.experiment_id}  {label}  dataset={e.dataset_version_id}")
        print(f"    {shlex.join(e.command)}")
    return 0


def cmd_exp_show(store: Store, exp_id: str) -> int:
    e = store.load_experiment(exp_id)
    ds = store.load_dataset(e.dataset_version_id)
    print(json_pretty(e.to_json()))
    print("\n--- reproduce ---")
    print(f"  export DVS_DATASET_VERSION={e.dataset_version_id}")
    print(f"  cd {e.workdir}")
    print(f"  # dataset root for that version: {ds.root}")
    print(f"  {shlex.join(e.command)}")
    return 0


def cmd_exp_verify(store: Store, exp_id: str) -> int:
    """Re-check that current directory still matches recorded dataset hash."""
    e = store.load_experiment(exp_id)
    ds = store.load_dataset(e.dataset_version_id)
    root = Path(ds.root)
    if not root.is_dir():
        print(f"Dataset root missing: {root}", file=sys.stderr)
        return 2
    current = build_file_map(root)
    current_id = dataset_version_id(str(root), current)
    if current_id != ds.version_id:
        print(f"Mismatch: stored {ds.version_id} vs live {current_id}", file=sys.stderr)
        a, r, c = diff_manifests(ds.files, current)
        print(f"  added {len(a)}, removed {len(r)}, changed {len(c)}", file=sys.stderr)
        return 1
    print("OK — directory still matches dataset version")
    return 0


def json_pretty(obj: object) -> str:
    import json

    return json.dumps(obj, indent=2)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="dvs", description="Mini dataset & experiment versioning")
    p.add_argument(
        "-C",
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing .dvs (default: cwd)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Create .dvs store")

    ap_add = sub.add_parser("add", help="Snapshot a directory as a dataset version")
    ap_add.add_argument("path", type=Path)
    ap_add.add_argument("--tag", default=None)

    sub.add_parser("list", help="List dataset versions")

    ap_show = sub.add_parser("show", help="Show manifest JSON for a dataset version")
    ap_show.add_argument("version_id")

    ap_diff = sub.add_parser("diff", help="Diff two dataset versions")
    ap_diff.add_argument("a")
    ap_diff.add_argument("b")

    ap_e = sub.add_parser("exp", help="Experiments")
    esub = ap_e.add_subparsers(dest="exp_cmd", required=True)

    ap_ec = esub.add_parser("create", help="Record an experiment tied to a dataset version")
    ap_ec.add_argument("--dataset", required=True)
    ap_ec.add_argument("--name", default=None)
    ap_ec.add_argument("--notes", default=None)
    ap_ec.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    # Allow `dvs exp create --dataset X -- python train.py` → command may start with empty if user forgets --
    ap_el = esub.add_parser("list", help="List experiments")
    ap_es = esub.add_parser("show", help="Show experiment + reproduce hints")
    ap_es.add_argument("experiment_id")
    ap_ev = esub.add_parser("verify", help="Check dataset directory still matches snapshot")
    ap_ev.add_argument("experiment_id")

    args = p.parse_args(argv)
    store = Store(args.project_root.resolve())

    if args.cmd == "init":
        return cmd_init(store)

    if not store.is_initialized():
        print("Run `dvs init` first.", file=sys.stderr)
        return 2

    if args.cmd == "add":
        return cmd_add(store, args.path, args.tag)
    if args.cmd == "list":
        return cmd_list_datasets(store)
    if args.cmd == "show":
        return cmd_show_dataset(store, args.version_id)
    if args.cmd == "diff":
        return cmd_diff(store, args.a, args.b)

    if args.cmd == "exp":
        if args.exp_cmd == "create":
            cmd_list = [c for c in args.command if c != ""]
            if cmd_list and cmd_list[0] == "--":
                cmd_list = cmd_list[1:]
            if not cmd_list:
                print("Provide a command, e.g. dvs exp create --dataset ID -- python train.py", file=sys.stderr)
                return 2
            return cmd_exp_create(store, args.dataset, cmd_list, args.name, args.notes)
        if args.exp_cmd == "list":
            return cmd_exp_list(store)
        if args.exp_cmd == "show":
            return cmd_exp_show(store, args.experiment_id)
        if args.exp_cmd == "verify":
            return cmd_exp_verify(store, args.experiment_id)

    return 1


def run() -> None:
    """Installed as the `dvs` console script (uv / pip)."""
    raise SystemExit(main())
