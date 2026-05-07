"""
hf_sync.py

HF Hub backup of training runs, designed for ephemeral compute (shared
clusters where local disk is wiped on disconnect). One repo holds many
runs as subfolders; on a fresh machine you fetch by run name and the
helper restores the local layout.

Layout convention (for a repo created via push_run_artifacts):
    hf://<repo_id>/
        <run_name>/
            <run_name>.log              # training stdout
            adapter/                    # final LoRA at best ckpt
            merged/                     # merged base+adapter
            checkpoint-{N}/...          # mid-training ckpts (optional)

Two ways to use this:

1. During training — attach HFCheckpointSyncCallback so each Trainer save
   is uploaded as it lands. Survives mid-run preemption. Failures are
   logged as warnings, never raised (training keeps going).

2. After training — call push_run_artifacts(...) once for the final
   adapter/, merged/, and the run log file (which sits OUTSIDE the
   trainer's output_dir in this project).

To fetch on a fresh machine:
    python train/hf_sync.py fetch \\
        --repo_id eenderyang/onerec-209b-runs \\
        --run_name orpo_5k \\
        --local_dir runs/

Or programmatically:
    from hf_sync import ensure_local
    merged_path = ensure_local(
        "runs/orpo_5k/merged",
        repo_id="eenderyang/onerec-209b-runs",
        repo_subpath="orpo_5k/merged",
    )

The ensure_local helper short-circuits when the local path exists, so
it's safe to call unconditionally at every script entry — the same
pattern as transformers' AutoModel cache.

Auth: reads ~/.cache/huggingface/token via huggingface_hub. No env
manipulation needed.

Important env-var note: HF_HUB_OFFLINE=1 blocks ALL Hub traffic
(including upload). The training scripts that opt into pushing must
NOT set HF_HUB_OFFLINE=1. TRANSFORMERS_OFFLINE=1 is fine to keep —
it only suppresses transformers' Hub retry on local-path loads.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_OFFLINE_TRUTHY = ("1", "true", "TRUE", "on", "ON", "yes", "YES")


def _check_hub_online() -> Optional[str]:
    """Return None if Hub access is plausibly available, else a reason string.

    huggingface_hub treats TRANSFORMERS_OFFLINE as a synonym for HF_HUB_OFFLINE
    (see hf_hub constants.py: ``HF_HUB_OFFLINE = is_true(HF_HUB_OFFLINE or
    TRANSFORMERS_OFFLINE)``). We check both so the failure mode is a clear
    pre-flight error, not a deep huggingface_hub traceback inside
    snapshot_download / upload_folder.
    """
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var, "").strip() in _OFFLINE_TRUTHY:
            return (f"{var}=1 is set — Hub traffic disabled. Unset it "
                    f"(and HF_HUB_OFFLINE) before running with Hub features. "
                    f"Note: huggingface_hub treats either var as offline.")
    return None


def _make_api(repo_id: str, repo_type: str, private: bool):
    """Lazy import + repo create. Returns an HfApi handle."""
    from huggingface_hub import HfApi, create_repo
    create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    return HfApi()


# ---------------------------------------------------------------------------
# Trainer callback — fires on every checkpoint save
# ---------------------------------------------------------------------------


def _make_callback_class():
    """Lazy-define the callback so this module can be imported without
    transformers (e.g. by a CLI fetch tool on a stripped-down env)."""
    from transformers import TrainerCallback

    class HFCheckpointSyncCallback(TrainerCallback):
        """Uploads each ``checkpoint-N/`` folder to ``<repo_id>:<run_subpath>/``
        as soon as Trainer writes it to disk. Mirrors the local layout 1:1.

        Failures don't propagate — a network blip won't kill a multi-hour run.
        Failed uploads are visible in stdout and can be retried at end-of-run
        via :func:`push_run_artifacts`.
        """

        def __init__(self, repo_id: str, run_subpath: str, run_dir: str,
                     repo_type: str = "model", private: bool = True,
                     ignore_patterns: Optional[List[str]] = None):
            self.repo_id = repo_id
            self.run_subpath = run_subpath.strip("/")
            self.run_dir = Path(run_dir)
            self.repo_type = repo_type
            self.private = private
            self.ignore_patterns = list(ignore_patterns or [])
            self._api = None
            self._init_failed = False

        def _ensure_api(self):
            if self._api is not None or self._init_failed:
                return
            offline = _check_hub_online()
            if offline:
                warnings.warn(f"[hf-sync] {offline}")
                self._init_failed = True
                return
            try:
                self._api = _make_api(self.repo_id, self.repo_type, self.private)
                print(f"[hf-sync] target: {self.repo_id} "
                      f"(repo_type={self.repo_type}, private={self.private}); "
                      f"run subpath: {self.run_subpath}")
            except Exception as e:
                print(f"[hf-sync] WARNING: create_repo / HfApi init failed: "
                      f"{type(e).__name__}: {e} — uploads disabled for this run")
                self._init_failed = True

        def on_save(self, args, state, control, **kwargs):
            self._ensure_api()
            if self._api is None:
                return
            ckpt_name = f"checkpoint-{state.global_step}"
            ckpt_dir = self.run_dir / ckpt_name
            if not ckpt_dir.exists():
                # Trainer can call on_save before the directory is materialized
                # in some edge cases — skip silently rather than warn.
                return
            path_in_repo = f"{self.run_subpath}/{ckpt_name}"
            try:
                self._api.upload_folder(
                    folder_path=str(ckpt_dir),
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                    ignore_patterns=self.ignore_patterns or None,
                    commit_message=f"sync {ckpt_name} (step {state.global_step})",
                )
                print(f"[hf-sync] pushed {ckpt_name} → "
                      f"{self.repo_id}:{path_in_repo}")
            except Exception as e:
                print(f"[hf-sync] WARNING: upload of {ckpt_name} failed: "
                      f"{type(e).__name__}: {e} — continuing training")

    return HFCheckpointSyncCallback


def HFCheckpointSyncCallback(*args, **kwargs):
    """Public constructor — wraps the lazy class so callers don't have to
    care about deferred transformers import."""
    cls = _make_callback_class()
    return cls(*args, **kwargs)


# ---------------------------------------------------------------------------
# End-of-run push
# ---------------------------------------------------------------------------


def push_run_artifacts(repo_id: str, run_subpath: str, run_dir: str,
                       log_path: Optional[str] = None,
                       include_dirs: Iterable[str] = ("adapter", "merged"),
                       extra_files: Optional[Iterable[str]] = None,
                       repo_type: str = "model", private: bool = True,
                       ignore_patterns: Optional[List[str]] = None) -> None:
    """Push final-state artifacts after training finishes.

    Args:
        repo_id:        e.g. "eenderyang/onerec-209b-runs"
        run_subpath:    subfolder inside the repo, e.g. "orpo_5k"
        run_dir:        local trainer output_dir, e.g. "runs/orpo_5k"
        log_path:       optional training-stdout log path. Sits OUTSIDE
                        run_dir in this project's convention
                        (runs/<name>.log next to runs/<name>/).
        include_dirs:   which subdirs of run_dir to push. Checkpoints are
                        already streamed by the callback; this call covers
                        adapter/ and merged/ which only exist after train
                        finishes.
        extra_files:    additional standalone files to push (e.g. the eval
                        log and CSV produced after training). Each is
                        uploaded under run_subpath/ using its basename.
    """
    offline = _check_hub_online()
    if offline:
        print(f"[hf-sync] {offline}")
        return

    try:
        api = _make_api(repo_id, repo_type, private)
    except Exception as e:
        print(f"[hf-sync] push_run_artifacts: api init failed "
              f"({type(e).__name__}: {e}); skipping")
        return

    run_subpath = run_subpath.strip("/")
    run_dir = Path(run_dir)

    for d in include_dirs:
        sub = run_dir / d
        if not sub.exists():
            print(f"[hf-sync] skip {d}/: not found at {sub}")
            continue
        try:
            api.upload_folder(
                folder_path=str(sub),
                path_in_repo=f"{run_subpath}/{d}",
                repo_id=repo_id, repo_type=repo_type,
                ignore_patterns=ignore_patterns or None,
                commit_message=f"final {d}",
            )
            print(f"[hf-sync] pushed {sub} → {repo_id}:{run_subpath}/{d}")
        except Exception as e:
            print(f"[hf-sync] WARNING: failed to push {d}: "
                  f"{type(e).__name__}: {e}")

    files_to_push = []
    if log_path:
        files_to_push.append(log_path)
    if extra_files:
        files_to_push.extend(extra_files)

    for fpath in files_to_push:
        fp = Path(fpath)
        if not fp.exists():
            print(f"[hf-sync] skip {fp}: not found")
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=f"{run_subpath}/{fp.name}",
                repo_id=repo_id, repo_type=repo_type,
                commit_message=f"final {fp.name}",
            )
            print(f"[hf-sync] pushed file → "
                  f"{repo_id}:{run_subpath}/{fp.name}")
        except Exception as e:
            print(f"[hf-sync] WARNING: failed to push {fp.name}: "
                  f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Fetch helper — local-first, Hub fallback
# ---------------------------------------------------------------------------


def ensure_local(local_path: str, repo_id: str,
                 repo_subpath: Optional[str] = None,
                 repo_type: str = "model",
                 patterns: Optional[List[str]] = None,
                 force_download: bool = False) -> str:
    """Return ``local_path``, downloading from HF Hub if missing.

    The standard caching primitive for ephemeral compute: at the top of any
    script that needs a checkpoint, call this and you'll get a working path
    whether or not the local disk has been wiped.

    Args:
        local_path:     where the artifact should live locally, e.g.
                        "runs/orpo_5k/merged".
        repo_id:        the HF repo to fetch from on cache miss.
        repo_subpath:   subdir inside the repo to fetch. Defaults to the
                        relative path of local_path under the parent that
                        will become the snapshot root.
        patterns:       explicit allow_patterns override. Defaults to
                        ``[f"{repo_subpath}/*"]``.
        force_download: skip the local-exists check and refetch.
    """
    local = Path(local_path)
    if local.exists() and any(local.iterdir() if local.is_dir() else [True]) \
            and not force_download:
        return str(local)

    from huggingface_hub import snapshot_download

    if repo_subpath is None:
        # Heuristic: assume layout is "<run_root>/<run_name>/<artifact>" and
        # use everything after the last 'runs/' as the in-repo subpath.
        # Caller should pass repo_subpath explicitly to avoid this guess.
        parts = local.parts
        if "runs" in parts:
            i = parts.index("runs")
            repo_subpath = "/".join(parts[i + 1:])
        else:
            repo_subpath = local.name

    if patterns is None:
        patterns = [f"{repo_subpath.strip('/')}/*"]

    # snapshot_download into the parent so the in-repo path mirrors locally.
    # e.g. local=runs/orpo_5k/merged, repo_subpath=orpo_5k/merged
    #   → local_dir=runs/, allow_patterns=['orpo_5k/merged/*']
    #   → creates runs/orpo_5k/merged/...
    depth = repo_subpath.count("/") + 1
    snapshot_root = Path(local)
    for _ in range(depth):
        snapshot_root = snapshot_root.parent
    snapshot_root.mkdir(parents=True, exist_ok=True)

    print(f"[hf-sync] fetching {repo_id}:{repo_subpath} → {snapshot_root}/")
    snapshot_download(
        repo_id=repo_id, repo_type=repo_type,
        allow_patterns=patterns,
        local_dir=str(snapshot_root),
    )
    return str(local)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_fetch(args):
    """Fetch a whole run (or a specific subdir) from Hub into a local tree."""
    if args.subdir:
        target = Path(args.local_dir) / args.run_name / args.subdir
        repo_subpath = f"{args.run_name}/{args.subdir}"
    else:
        target = Path(args.local_dir) / args.run_name
        repo_subpath = args.run_name
    ensure_local(
        local_path=str(target),
        repo_id=args.repo_id,
        repo_subpath=repo_subpath,
        repo_type=args.repo_type,
        force_download=args.force,
    )
    print(f"[hf-sync] ready at {target}")


def _cli_push(args):
    push_run_artifacts(
        repo_id=args.repo_id,
        run_subpath=args.run_name,
        run_dir=args.run_dir,
        log_path=args.log_path,
        include_dirs=args.include,
        extra_files=args.extra_files,
        repo_type=args.repo_type,
        private=not args.public,
    )


def _cli_pull(args):
    """Whole-repo download (e.g. base model). Distinct from `fetch` which
    targets a subdir of a multi-run repo."""
    target = Path(args.local_path)
    if target.exists() and any(target.iterdir() if target.is_dir() else [True]) \
            and not args.force:
        print(f"[hf-sync] {target} already populated; pass --force to refetch")
        return
    offline = _check_hub_online()
    if offline:
        raise RuntimeError(offline)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[hf-sync] pulling {args.repo_id} → {target} "
          f"(repo_type={args.repo_type})")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(target),
        allow_patterns=args.allow_patterns,
        ignore_patterns=args.ignore_patterns,
    )
    print(f"[hf-sync] done")


def _build_parser():
    parser = argparse.ArgumentParser(prog="hf_sync")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser(
        "fetch",
        help="Download a run (or one of its subdirs) from HF Hub to local disk.",
    )
    p_fetch.add_argument("--repo_id", required=True)
    p_fetch.add_argument("--run_name", required=True,
                         help='Subfolder name in the repo, e.g. "orpo_5k"')
    p_fetch.add_argument("--subdir", default=None,
                         help='Optional inner dir, e.g. "merged" '
                              '(default: fetch the whole run)')
    p_fetch.add_argument("--local_dir", default="runs",
                         help='Local root for the run (default: "runs")')
    p_fetch.add_argument("--repo_type", default="model",
                         choices=["model", "dataset", "space"])
    p_fetch.add_argument("--force", action="store_true",
                         help="Refetch even if local files exist.")
    p_fetch.set_defaults(func=_cli_fetch)

    p_push = sub.add_parser(
        "push",
        help="Push final adapter/merged/log of an existing local run.",
    )
    p_push.add_argument("--repo_id", required=True)
    p_push.add_argument("--run_name", required=True)
    p_push.add_argument("--run_dir", required=True,
                        help='Local trainer output_dir, e.g. "runs/orpo_5k"')
    p_push.add_argument("--log_path", default=None,
                        help='Optional path to training stdout log.')
    p_push.add_argument("--include", nargs="*",
                        default=["adapter", "merged"],
                        help="Subdirs of --run_dir to push. Pass an empty "
                             "list (--include) to skip directories and only "
                             "push --log_path / --extra_files.")
    p_push.add_argument("--extra_files", nargs="*", default=None,
                        help="Additional standalone files to push under "
                             "<run_name>/<basename>.")
    p_push.add_argument("--repo_type", default="model",
                        choices=["model", "dataset", "space"])
    p_push.add_argument("--public", action="store_true",
                        help="Create as public repo (default: private).")
    p_push.set_defaults(func=_cli_push)

    p_pull = sub.add_parser(
        "pull",
        help="Whole-repo download (e.g. base model + tokenizer) into a local dir.",
    )
    p_pull.add_argument("--repo_id", required=True,
                        help='Source HF repo, e.g. "OpenOneRec/OneRec-1.7B"')
    p_pull.add_argument("--local_path", required=True,
                        help='Local directory to populate, e.g. "model/OneRec-1.7B"')
    p_pull.add_argument("--repo_type", default="model",
                        choices=["model", "dataset", "space"])
    p_pull.add_argument("--allow_patterns", nargs="*", default=None,
                        help="Optional glob allowlist (e.g. '*.safetensors' "
                             "'tokenizer*'). Default: pull everything.")
    p_pull.add_argument("--ignore_patterns", nargs="*", default=None,
                        help="Optional glob denylist (e.g. 'assets/*' "
                             "'*.png' to skip docs and previews).")
    p_pull.add_argument("--force", action="store_true",
                        help="Refetch even if target already populated.")
    p_pull.set_defaults(func=_cli_pull)

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
