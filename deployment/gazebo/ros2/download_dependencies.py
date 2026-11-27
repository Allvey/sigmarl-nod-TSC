"""Download apt's authenticated package selection concurrently before installation."""
import concurrent.futures
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

selection = subprocess.run(["apt-get", "--yes", "--no-install-recommends", "--download-only", "--print-uris", "install", *sys.argv[1:]],
                           check=True, capture_output=True, text=True)
packages = [shlex.split(line) for line in selection.stdout.splitlines() if line.startswith("'")]


def download(package):
    url, filename, size, checksum = package
    algorithm, digest = checksum.split(":", 1)
    algorithm = {"MD5Sum": "md5"}.get(algorithm, algorithm.lower())
    destination = Path("/var/cache/apt/archives") / filename
    if destination.name != filename:
        raise ValueError("Unexpected apt package filename")

    def verified(content):
        return len(content) == int(size) and hashlib.new(algorithm, content).hexdigest() == digest

    if destination.exists() and verified(destination.read_bytes()):
        return
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            if not verified(content):
                raise RuntimeError("Package size or checksum mismatch: " + filename)
            destination.write_bytes(content)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


print(f"Downloading {len(packages)} verified apt packages", flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for index, _ in enumerate(pool.map(download, packages), 1):
        if index % 50 == 0 or index == len(packages):
            print(f"Downloaded {index}/{len(packages)} packages", flush=True)
