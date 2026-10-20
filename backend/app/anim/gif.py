"""动画导出为 GIF（自带编码器）。

为什么不用 PIL 的 `save("GIF")`：
- PIL 保存 GIF 时会为每帧**建局部调色表**并重排索引，而透明索引是
  相对于调色表的。我们控制不了它重排成什么样，于是透明索引会指错位置
  ——实测过：皮肤色的像素被当成透明，帧里出现大块洋红。
  （试过「所有帧调色板相同就不重排」「洋红放索引 0/254/255」，
  都不管用——PIL 照样重排。）
- 逐帧独立量化更糟：每帧调色板不同，播放时颜色**闪烁**
  （同一块皮肤在两帧里是两种色）。

所以这里自己写编码器：一张**全局调色板**、一个**全局透明索引**、
每帧 LZW 压缩。GIF89a 的这几部分都不复杂，写一遍换来完全可控。

GIF 的固有限制（写清楚，免得当成 bug）：
- 最多 256 色；
- 透明只有 1 bit（要么全透明要么不透明，半透明会被阈值化）；
- 逐帧栅格化，没有半透明混合。
需要保真请用 PNG 序列或 Godot 骨架包。
"""
from __future__ import annotations

import struct
from typing import List, Optional, Sequence, Tuple

from PIL import Image

# 透明像素用的哨兵色：必须是**内容里不可能出现**的颜色，
# 否则真实的洋红像素会被误判成透明。
_SENTINEL = (255, 0, 255)


def _lzw_encode(indices: bytes, min_code_size: int) -> bytes:
    """GIF 用的 LZW 压缩（变长码，LSB 优先）。

    与 PNG 的 LZW 不同：GIF 是「先输出码再输出数据」，
    而且**码表满了必须发清表码**——不处理这点，文件解不出来。
    """
    clear_code = 1 << min_code_size
    end_code = clear_code + 1
    code_size = min_code_size + 1
    next_code = end_code + 1
    table: List[bytes] = [bytes([i]) for i in range(clear_code)] + [b"", b""]
    out = bytearray()
    bit_buf = 0
    bit_cnt = 0

    def emit(code: int) -> None:
        nonlocal bit_buf, bit_cnt
        bit_buf |= code << bit_cnt
        bit_cnt += code_size
        while bit_cnt >= 8:
            out.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_cnt -= 8

    emit(clear_code)
    if not indices:
        emit(end_code)
        if bit_cnt:
            out.append(bit_buf & 0xFF)
        return bytes(out)

    prefix = b""
    for byte in indices:
        cur = prefix + bytes([byte])
        if cur in table:
            prefix = cur
            continue
        emit(table.index(prefix))
        if next_code < 4096:
            table.append(cur)
            next_code += 1
            if next_code > (1 << code_size) and code_size < 12:
                code_size += 1
        else:
            # 码表满：发清表码重来
            emit(clear_code)
            table = [bytes([i]) for i in range(clear_code)] + [b"", b""]
            next_code = end_code + 1
            code_size = min_code_size + 1
        prefix = bytes([byte])
    if prefix:
        emit(table.index(prefix))
    emit(end_code)
    if bit_cnt:
        out.append(bit_buf & 0xFF)
    return bytes(out)


def _blocks(data: bytes) -> bytes:
    """GIF 的数据子块格式：每块 ≤255 字节，前置长度字节，以 0 结尾。"""
    out = bytearray()
    for i in range(0, len(data), 255):
        chunk = data[i:i + 255]
        out.append(len(chunk))
        out.extend(chunk)
    out.append(0)
    return bytes(out)


def _build_palette(frames: Sequence[Image.Image], w: int, h: int,
                   max_colors: int) -> Tuple[List[Tuple[int, int, int]],
                                           List[bytes], int]:
    """全局调色板 + 每帧索引图 + 透明索引。

    调色板从**所有帧拼成的大图**统一量化——各帧共用一张表，颜色不闪。
    透明像素在量化前替换成哨兵色，量化后按哨兵色找回索引。
    """
    flats: List[Image.Image] = []
    masks: List[Image.Image] = []
    for f in frames:
        rgba = f if f.mode == "RGBA" else f.convert("RGBA")
        if rgba.size != (w, h):
            rgba = rgba.resize((w, h), Image.NEAREST)
        a = rgba.getchannel("A")
        # GIF 透明只有 1 bit：半透明按阈值二值化
        masks.append(a.point(lambda v: 255 if v >= 128 else 0, mode="L"))
        bg = Image.new("RGBA", (w, h), _SENTINEL + (255,))
        bg.alpha_composite(rgba)
        flats.append(bg.convert("RGB"))

    strip = Image.new("RGB", (w * len(flats), h))
    for i, im in enumerate(flats):
        strip.paste(im, (i * w, 0))
    # 内容最多用 255 色，留一个索引给哨兵
    n_content = max(2, min(255, max_colors - 1))
    pal_img = strip.quantize(colors=n_content, method=Image.MEDIANCUT)
    pal = pal_img.getpalette() or []
    colors: List[Tuple[int, int, int]] = []
    for i in range(len(pal) // 3):
        colors.append((pal[i * 3], pal[i * 3 + 1], pal[i * 3 + 2]))

    # 哨兵色在调色板里的索引。MEDIANCUT 可能把它并到别的颜色上，
    # 那时找不到——这种情况退化用「黑色」（透明区会显示黑色，见 README 限制）。
    sent_idx: Optional[int] = None
    for i, c in enumerate(colors):
        if c[0] > 200 and c[1] < 60 and c[2] > 200:
            sent_idx = i
            break

    index_frames: List[bytes] = []
    for i, im in enumerate(flats):
        q = im.quantize(palette=pal_img, dither=Image.FLOYDSTEINBERG)
        if sent_idx is not None:
            m = masks[i].point(lambda v: 255 if v < 128 else 0, mode="L")
            q.paste(sent_idx, mask=m)
        index_frames.append(bytes(q.tobytes()))

    if sent_idx is None:
        # 调色板里没有哨兵色：给透明区找一个「内容用得少」的索引。
        # 用出现次数最少的那个——它多半是背景色，透明区显示成它最不显眼。
        counts: dict = {}
        for idx_map in index_frames:
            for v in idx_map:
                counts[v] = counts.get(v, 0) + 1
        if counts:
            sent_idx = min(counts, key=lambda k: counts[k])
        else:
            sent_idx = 0
        for i in range(len(index_frames)):
            m = masks[i].point(lambda v: 255 if v < 128 else 0, mode="L")
            index_frames[i] = bytes(
                sent_idx if m.getpixel((k % w, k // w)) else v
                for k, v in enumerate(index_frames[i]))
        colors.append((0, 0, 0))
    else:
        colors.append(_SENTINEL)

    return colors, index_frames, sent_idx


def to_gif(frames: Sequence[Image.Image],
           *,
           duration_ms: int = 100,
           loop: int = 0,
           max_colors: int = 256) -> bytes:
    """把一组 RGBA 帧编码成 GIF 字节。

    duration_ms 每帧时长（毫秒）；loop=0 表示无限循环。
    """
    if not frames:
        raise ValueError("没有可导出的帧")
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    colors, index_frames, trans = _build_palette(frames, w, h, max_colors)

    # 调色板大小必须是 2 的幂（GIF 规范）
    n = 1
    while n < len(colors):
        n *= 2
    pal_bytes = bytearray()
    for i in range(n):
        r, g, b = colors[i] if i < len(colors) else (0, 0, 0)
        pal_bytes.extend((r, g, b))

    out = bytearray()
    out.extend(b"GIF89a")
    out.extend(struct.pack("<HH", w, h))
    bits = max(1, (n - 1).bit_length())
    # 全局调色板标志 + 色深；背景索引 0、像素纵横比 0
    out.extend(bytes([0xF0 | (bits - 1), 0, 0]))
    out.extend(pal_bytes)
    # NETSCAPE 循环扩展
    out.extend(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01")
    out.extend(struct.pack("<H", int(loop)))
    out.extend(b"\x00")

    min_code = max(2, bits)
    for idx in index_frames:
        # 图形控制扩展：disposal=2（恢复背景）+ 透明标志 + 帧延时
        out.extend(b"\x21\xF9\x04")
        out.extend(bytes([0b0000_0101]))
        out.extend(struct.pack("<H", max(20, int(duration_ms))))
        out.extend(bytes([trans & 0xFF, 0]))
        # 图像描述符（无局部调色板）
        out.extend(b"\x2C")
        out.extend(struct.pack("<HHHH", 0, 0, w, h))
        out.extend(bytes([0]))
        out.extend(bytes([min_code]))
        out.extend(_blocks(_lzw_encode(idx, min_code)))

    out.extend(b"\x3B")   # trailer
    return bytes(out)
