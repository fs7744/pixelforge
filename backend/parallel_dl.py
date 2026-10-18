"""分块并发下载单文件（走代理时比单连接快 2-4 倍）。

用法：
  python backend/parallel_dl.py <repo> <file> <dest> [chunks]
例如：
  python backend/parallel_dl.py Lykon/dreamshaper-8 vae/diffusion_pytorch_model.fp16.safetensors models/dreamshaper-8/vae/x.safetensors

代理：设环境变量 HTTPS_PROXY=http://127.0.0.1:7890
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

PROXY = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""


def _opener() -> urllib.request.OpenerDirector:
    if PROXY:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
    return urllib.request.build_opener()


def _size(url: str) -> int:
    """用 Range 探测总大小。

    代理（clash 之类）常常不支持 HEAD，会回 502；
    发一个 bytes=0-0 的 GET，从 Content-Range 里读总量最稳。
    """
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with _opener().open(req, timeout=30) as r:
        cr = r.headers.get("Content-Range") or ""
        if "/" in cr:
            try:
                return int(cr.rsplit("/", 1)[1])
            except ValueError:
                pass
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else 0


def download(repo: str, path: str, dest: str, chunks: int = 6) -> None:
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    total = _size(url)
    if total <= 0:
        raise RuntimeError("拿不到文件大小")
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == total:
        print(f"skip {dest} ({total/1024**2:.1f}MB 已存在)")
        return

    step = total // chunks
    parts = [(i * step, total - 1 if i == chunks - 1 else (i + 1) * step - 1) for i in range(chunks)]
    tmp = [out.with_suffix(out.suffix + f".p{i}") for i in range(chunks)]

    def grab(i: int, start: int, end: int) -> None:
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with _opener().open(req, timeout=60) as r, open(tmp[i], "wb") as f:
            while True:
                buf = r.read(262144)
                if not buf:
                    break
                f.write(buf)

    t0 = time.time()
    threads = [threading.Thread(target=grab, args=(i, s, e)) for i, (s, e) in enumerate(parts)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(out, "wb") as f:
        for p in tmp:
            f.write(p.read_bytes())
            p.unlink()
    size = out.stat().st_size
    dt = time.time() - t0
    print(f"ok {dest} {size/1024**2:.1f}MB in {dt:.0f}s ({size/1024/dt:.0f} KB/s)")
    if size != total:
        raise RuntimeError(f"大小不符 {size} != {total}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)
    download(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 6)
