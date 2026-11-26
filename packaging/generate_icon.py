"""Generate the multi-resolution Windows .ico asset using Python stdlib only."""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ICON_SIZES = (16, 32, 48, 256)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "geomag_analyzer.ico"

# Colors matching frontend/index.html SVG favicon (#0f172a background, #38bdf8 trace)
BG_RGB = (15, 23, 42)
BORDER_RGB = (30, 41, 59)
TRACE_RGB = (56, 189, 248)
GLOW_RGB = (14, 165, 233)

# Polyline path in normalized [0, 32] coordinates from frontend/index.html:
# M4 17 h5 l3-8 l4 14 l4-10 l3 4 h5
TRACE_POINTS: tuple[tuple[float, float], ...] = (
    (4.0, 17.0),
    (9.0, 17.0),
    (12.0, 9.0),
    (16.0, 23.0),
    (20.0, 13.0),
    (23.0, 17.0),
    (28.0, 17.0),
)


def _dist_to_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _rounded_rect_sdf(px: float, py: float, size: float, radius: float) -> float:
    half = size * 0.5
    qx = abs(px - half) - (half - radius)
    qy = abs(py - half) - (half - radius)
    out_dist = math.hypot(max(qx, 0.0), max(qy, 0.0))
    in_dist = min(max(qx, qy), 0.0)
    return out_dist + in_dist - radius


def _blend_rgb(
    base: tuple[int, int, int],
    fg: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    a = max(0.0, min(1.0, alpha))
    return (
        int(round(base[0] * (1.0 - a) + fg[0] * a)),
        int(round(base[1] * (1.0 - a) + fg[1] * a)),
        int(round(base[2] * (1.0 - a) + fg[2] * a)),
    )


def render_rgba_pixels(size: int) -> bytes:
    """Render a crisp anti-aliased RGBA buffer (top-down row major) for a given icon size."""
    scale = size / 32.0
    # Slightly thicker stroke at 16px so the waveform stays crisp on small taskbars
    stroke_radius = max(1.05, (1.35 if size <= 16 else 1.25) * scale)
    corner_radius = 7.0 * scale
    aa = 0.85
    scaled_pts = [(x * scale, y * scale) for x, y in TRACE_POINTS]
    segments = list(zip(scaled_pts[:-1], scaled_pts[1:], strict=True))

    buf = bytearray(size * size * 4)
    idx = 0
    for y in range(size):
        py = y + 0.5
        for x in range(size):
            px = x + 0.5
            sdf = _rounded_rect_sdf(px, py, float(size), corner_radius)
            bg_alpha = max(0.0, min(1.0, 0.5 - sdf / aa))
            if bg_alpha <= 0.0:
                buf[idx : idx + 4] = b"\x00\x00\x00\x00"
                idx += 4
                continue

            rgb = BG_RGB
            # Subtle inner rim highlight
            rim_dist = abs(sdf + max(0.75, 0.5 * scale))
            if rim_dist < max(0.8, 0.5 * scale):
                rim_a = max(0.0, 1.0 - rim_dist / max(0.8, 0.5 * scale)) * 0.75
                rgb = _blend_rgb(rgb, BORDER_RGB, rim_a)

            min_seg_d = min(
                _dist_to_segment(px, py, a[0], a[1], b[0], b[1])
                for a, b in segments
            )
            glow_radius = stroke_radius + max(0.9, 1.1 * scale)
            if min_seg_d < glow_radius:
                glow_a = max(0.0, 1.0 - min_seg_d / glow_radius) * 0.28
                rgb = _blend_rgb(rgb, GLOW_RGB, glow_a)

            trace_alpha = max(0.0, min(1.0, (stroke_radius + 0.5 * aa - min_seg_d) / aa))
            if trace_alpha > 0.0:
                rgb = _blend_rgb(rgb, TRACE_RGB, trace_alpha)

            buf[idx] = rgb[0]
            buf[idx + 1] = rgb[1]
            buf[idx + 2] = rgb[2]
            buf[idx + 3] = int(round(bg_alpha * 255.0))
            idx += 4
    return bytes(buf)


def _encode_bmp_dib_entry(size: int, rgba: bytes) -> bytes:
    """Encode 32-bit BGRA DIB + 1-bpp AND mask for <=48px ICO entries."""
    xor_rows = bytearray()
    and_stride = ((size + 31) // 32) * 4
    and_rows = bytearray()

    for y in range(size - 1, -1, -1):
        row_start = y * size * 4
        and_row = bytearray(and_stride)
        for x in range(size):
            p = row_start + x * 4
            r, g, b, a = rgba[p], rgba[p + 1], rgba[p + 2], rgba[p + 3]
            xor_rows.extend((b, g, r, a))
            if a < 128:
                and_row[x // 8] |= 1 << (7 - (x % 8))
        and_rows.extend(and_row)

    xor_bytes = bytes(xor_rows)
    and_bytes = bytes(and_rows)
    header = struct.pack(
        "<IIIHHIIIIII",
        40,  # biSize
        size,  # biWidth
        size * 2,  # biHeight (XOR + AND)
        1,  # biPlanes
        32,  # biBitCount
        0,  # biCompression = BI_RGB
        len(xor_bytes) + len(and_bytes),  # biSizeImage
        0,  # biXPelsPerMeter
        0,  # biYPelsPerMeter
        0,  # biClrUsed
        0,  # biClrImportant
    )
    return header + xor_bytes + and_bytes


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _encode_png_entry(size: int, rgba: bytes) -> bytes:
    """Encode 32-bit RGBA PNG payload for 256x256 ICO entry."""
    raw_scanlines = bytearray()
    row_bytes = size * 4
    for y in range(size):
        raw_scanlines.append(0)  # Filter type 0 (None)
        raw_scanlines.extend(rgba[y * row_bytes : (y + 1) * row_bytes])

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw_scanlines), level=9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def build_ico_bytes(sizes: tuple[int, ...] = ICON_SIZES) -> bytes:
    """Build complete multi-resolution Windows .ico binary."""
    entries: list[tuple[int, bytes]] = []
    for sz in sizes:
        rgba = render_rgba_pixels(sz)
        payload = _encode_png_entry(sz, rgba) if sz >= 256 else _encode_bmp_dib_entry(sz, rgba)
        entries.append((sz, payload))

    header = struct.pack("<HHH", 0, 1, len(entries))
    offset = 6 + 16 * len(entries)
    dir_entries = bytearray()
    payloads = bytearray()
    for sz, payload in entries:
        w_byte = 0 if sz >= 256 else sz
        h_byte = 0 if sz >= 256 else sz
        dir_entries.extend(
            struct.pack(
                "<BBBBHHII",
                w_byte,
                h_byte,
                0,  # bColorCount
                0,  # bReserved
                1,  # wPlanes
                32,  # wBitCount
                len(payload),
                offset,
            )
        )
        payloads.extend(payload)
        offset += len(payload)
    return header + bytes(dir_entries) + bytes(payloads)


def generate_icon_file(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Write multi-resolution Windows .ico asset to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(build_ico_bytes())
    return out


if __name__ == "__main__":
    written = generate_icon_file()
    print(f"Generated {written} ({written.stat().st_size} bytes)")
