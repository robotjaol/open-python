"""Download files with checksum check."""

import hashlib
import urllib.request
from pathlib import Path


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url, destination, expected_sha256=None):
    """Fetch url once, verify hash."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    if expected_sha256:
        actual = sha256_of(destination)
        if actual != expected_sha256:
            destination.unlink()
            raise ValueError(f"checksum mismatch for {url}: {actual}")
    return destination
