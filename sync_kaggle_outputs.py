#!/usr/bin/env python3
"""
Sync Kaggle notebook outputs into a local directory.

Example:
  python sync_kaggle_outputs.py \
      --kernel mustafa-munir/my-notebook \
      --dest /Users/mustafa.munir/Personal/research

You can also pass a full Kaggle URL for --kernel:
  https://www.kaggle.com/code/<owner>/<slug>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
import tarfile
import zipfile
from pathlib import Path
from urllib3.exceptions import InsecureRequestWarning


def _load_dotenv(dotenv_path: Path, *, override: bool = True) -> None:
    """Load simple KEY=VALUE pairs from .env into os.environ."""
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            if override:
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)
    # Common alias people keep in .env
    if "KAGGLE_API_KEY" in os.environ:
        if override or "KAGGLE_KEY" not in os.environ:
            os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_KEY"]


def _parse_kernel_and_session(raw: str) -> tuple[str, int]:
    raw = raw.strip()
    if "kaggle.com" not in raw:
        return raw, 0
    m = re.search(r"kaggle\.com/code/([^/]+)/([^/?#]+)", raw)
    if not m:
        raise ValueError(f"Could not parse kernel ref from URL: {raw}")
    kernel_ref = f"{m.group(1)}/{m.group(2)}"
    session_match = re.search(r"/edit/run/(\d+)", raw)
    session_id = int(session_match.group(1)) if session_match else 0
    return kernel_ref, session_id


def _file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _same_file(src: Path, dst: Path) -> bool:
    if not dst.exists() or not dst.is_file():
        return False
    if src.stat().st_size != dst.stat().st_size:
        return False
    return _file_hash(src) == _file_hash(dst)


def _copy_tree_sync(src_root: Path, dst_root: Path, delete: bool) -> tuple[int, int, int]:
    copied = 0
    skipped = 0
    deleted = 0
    src_files: set[Path] = set()

    for src in src_root.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        src_files.add(rel)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if _same_file(src, dst):
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    if delete and dst_root.exists():
        for dst in dst_root.rglob("*"):
            if not dst.is_file():
                continue
            rel = dst.relative_to(dst_root)
            if rel not in src_files:
                dst.unlink()
                deleted += 1

    return copied, skipped, deleted


def _extract_archive(archive_path: Path, extract_to: Path) -> bool:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_to)
        return True
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_to)
        return True
    return False


def _download_output_bundle(
    kernel_ref: str, owner_slug: str, kernel_slug: str, tmp_path: Path, verify_tls: bool
) -> Path | None:
    import requests

    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if not username or not key:
        return None

    bundle_url = f"https://www.kaggle.com/api/v1/kernels/output/download/{owner_slug}/{kernel_slug}"
    out_archive = tmp_path / "kernel_output_bundle"
    resp = requests.get(
        bundle_url,
        auth=(username, key),
        stream=True,
        timeout=120,
        verify=verify_tls,
        allow_redirects=True,
    )
    if resp.status_code != 200:
        print(
            f"Bundle download unavailable ({resp.status_code}) for {kernel_ref}: "
            f"{(resp.text or '').strip()[:300]}"
        )
        return None

    # Best effort: infer extension from content-disposition/content-type.
    cdisp = resp.headers.get("content-disposition", "")
    ctype = resp.headers.get("content-type", "")
    if ".zip" in cdisp or "zip" in ctype:
        out_archive = out_archive.with_suffix(".zip")
    elif ".tar" in cdisp or "tar" in ctype or "gzip" in ctype:
        out_archive = out_archive.with_suffix(".tar.gz")

    with out_archive.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    extract_dir = tmp_path / "bundle_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not _extract_archive(out_archive, extract_dir):
        return None
    print(f"Downloaded output bundle for {kernel_ref}")
    return extract_dir


def _download_session_output_zip(
    session_id: int, tmp_path: Path, verify_tls: bool
) -> Path | None:
    import requests

    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if not username or not key:
        return None

    zip_url = f"https://www.kaggle.com/api/v1/kernels/output/download_zip/{session_id}"
    out_zip = tmp_path / f"kernel_session_{session_id}.zip"
    resp = requests.get(
        zip_url,
        auth=(username, key),
        stream=True,
        timeout=120,
        verify=verify_tls,
        allow_redirects=True,
    )
    if resp.status_code != 200:
        print(
            f"Session zip unavailable ({resp.status_code}) for session {session_id}: "
            f"{(resp.text or '').strip()[:300]}"
        )
        return None

    with out_zip.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    extract_dir = tmp_path / f"session_{session_id}_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not _extract_archive(out_zip, extract_dir):
        return None
    print(f"Downloaded output zip for session {session_id}")
    return extract_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download Kaggle kernel outputs and sync to local folder."
    )
    p.add_argument(
        "--kernel",
        required=True,
        help="Kernel ref like 'owner/slug' or full https://www.kaggle.com/code/... URL",
    )
    p.add_argument(
        "--session-id",
        type=int,
        default=0,
        help="Optional Kaggle run/session id from URL (.../edit/run/<id>) to fetch that exact run output zip.",
    )
    p.add_argument(
        "--dest",
        default=".",
        help="Local destination root (default: current directory).",
    )
    p.add_argument(
        "--delete",
        action="store_true",
        help="Delete local files not present in Kaggle output.",
    )
    p.add_argument(
        "--ca-bundle",
        default="",
        help="Path to corporate CA PEM bundle (sets REQUESTS_CA_BUNDLE and SSL_CERT_FILE).",
    )
    p.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (last resort).",
    )
    p.add_argument(
        "--list-files",
        action="store_true",
        help="Print files returned by Kaggle before syncing.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    kernel_ref, inferred_session_id = _parse_kernel_and_session(args.kernel)
    session_id = args.session_id or inferred_session_id
    kernel_owner = kernel_ref.split("/", 1)[0] if "/" in kernel_ref else ""
    kernel_slug = kernel_ref.split("/", 1)[1] if "/" in kernel_ref else kernel_ref
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    _load_dotenv(Path(".env"), override=True)

    ca_bundle = args.ca_bundle or os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if ca_bundle:
        ca_path = Path(ca_bundle).expanduser().resolve()
        if not ca_path.exists():
            print(f"CA bundle not found: {ca_path}")
            return 2
        os.environ["REQUESTS_CA_BUNDLE"] = str(ca_path)
        os.environ["SSL_CERT_FILE"] = str(ca_path)

    if args.insecure:
        # Last resort for corporate MITM environments where root CA cannot be provided.
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        try:
            import requests
            import urllib3

            old_merge_environment_settings = requests.Session.merge_environment_settings

            def _merge_environment_settings_noverify(self, url, proxies, stream, verify, cert):
                settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
                settings["verify"] = False
                return settings

            requests.Session.merge_environment_settings = _merge_environment_settings_noverify
            urllib3.disable_warnings(InsecureRequestWarning)
        except Exception:
            pass
    verify_tls = not args.insecure

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # pragma: no cover
        print("kaggle package is not installed or broken.")
        print("Install it with: pip install kaggle")
        print(f"Details: {e}")
        return 2

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print("Kaggle authentication failed.")
        print("Make sure ~/.kaggle/kaggle.json exists or KAGGLE_USERNAME/KAGGLE_KEY are set.")
        print(f"Details: {e}")
        return 2
    authed_user = (api.get_config_value(api.CONFIG_NAME_USER) or "").strip()
    if authed_user:
        print(f"Authenticated Kaggle user: {authed_user}")

    with tempfile.TemporaryDirectory(prefix="kaggle-output-") as tmp:
        tmp_path = Path(tmp)
        if session_id:
            session_dir = _download_session_output_zip(
                session_id=session_id, tmp_path=tmp_path, verify_tls=verify_tls
            )
            if session_dir is not None:
                if args.list_files:
                    print("Files returned by Kaggle (session zip):")
                    for p in sorted(session_dir.rglob("*")):
                        if p.is_file():
                            print("-", p.relative_to(session_dir))
                copied, skipped, deleted = _copy_tree_sync(
                    src_root=session_dir, dst_root=dest, delete=args.delete
                )
                print(f"Kernel: {kernel_ref}")
                print(f"Session ID: {session_id}")
                print(f"Destination: {dest}")
                print(f"Copied/updated: {copied}")
                print(f"Unchanged: {skipped}")
                print(f"Deleted: {deleted}")
                return 0
            print(f"Session output zip not available for session {session_id}; falling back.")

        bundle_dir = _download_output_bundle(
            kernel_ref=kernel_ref,
            owner_slug=kernel_owner,
            kernel_slug=kernel_slug,
            tmp_path=tmp_path,
            verify_tls=verify_tls,
        )

        if bundle_dir is not None:
            if args.list_files:
                print("Files returned by Kaggle (bundle):")
                for p in sorted(bundle_dir.rglob("*")):
                    if p.is_file():
                        print("-", p.relative_to(bundle_dir))
            copied, skipped, deleted = _copy_tree_sync(
                src_root=bundle_dir, dst_root=dest, delete=args.delete
            )
            print(f"Kernel: {kernel_ref}")
            print(f"Destination: {dest}")
            print(f"Copied/updated: {copied}")
            print(f"Unchanged: {skipped}")
            print(f"Deleted: {deleted}")
            return 0

        try:
            api.kernels_output(kernel=kernel_ref, path=str(tmp_path), quiet=False, force=True)
        except Exception as e:
            print(f"Failed to download outputs for '{kernel_ref}'.")
            print(f"Details: {e}")
            msg = str(e).lower()
            if "403" in msg or "forbidden" in msg:
                print("\nLikely causes:")
                if authed_user and kernel_owner and authed_user != kernel_owner:
                    print(f"- Authenticated as '{authed_user}' but kernel owner is '{kernel_owner}'.")
                print("- Kernel outputs are not accessible to this token (private/draft/no permission).")
                print("- In Kaggle UI, open the notebook and click 'Save Version' after a successful run.")
                print("- Then retry this command with the same owner/slug.")
            return 1

        copied, skipped, deleted = _copy_tree_sync(
            src_root=tmp_path, dst_root=dest, delete=args.delete
        )

    print(f"Kernel: {kernel_ref}")
    print(f"Destination: {dest}")
    print(f"Copied/updated: {copied}")
    print(f"Unchanged: {skipped}")
    print(f"Deleted: {deleted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
