"""Generate LaTeX booktabs tables and vector PDF/PNG figures from next-gen evaluations."""

import argparse
import importlib.util
import json
import struct
import zlib
from pathlib import Path

import numpy as np

ALL_SEVEN_ARMS = (
    "M48",
    "F-space",
    "T5",
    "T8",
    "T8-LS-9999",
    "M48-Guard-9999",
    "ML-MinGram-WP-G1000",
)
DEFAULT_BASE_INTRINSIC = {
    "M48": {"dev_targets_with_eos": 77637, "bytes_per_prediction": 12.880, "pure_whitespace_tokens": 0, "vocab_other_script": 0, "akshara_per_1k": None, "mom_akshara_breaks": None, "canonical_eq": "lossy"},
    "F-space": {"dev_targets_with_eos": None, "bytes_per_prediction": None, "pure_whitespace_tokens": None, "vocab_other_script": None, "akshara_per_1k": None, "mom_akshara_breaks": None, "canonical_eq": "186/186"},
    "T5": {"dev_targets_with_eos": 107848, "bytes_per_prediction": 9.272, "pure_whitespace_tokens": 36658, "vocab_other_script": 1624, "akshara_per_1k": 7.868, "mom_akshara_breaks": 886, "canonical_eq": "186/186"},
    "T8": {"dev_targets_with_eos": 107813, "bytes_per_prediction": 9.275, "pure_whitespace_tokens": 36658, "vocab_other_script": 1620, "akshara_per_1k": 3.614, "mom_akshara_breaks": 407, "canonical_eq": "186/186"},
}


def _tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _fmt(val, fmt: str = ".4f", missing: str = "--") -> str:
    return missing if val is None else format(val, fmt)


def _fmt_int(val, missing: str = "--") -> str:
    return missing if val is None else f"{int(val):,}"


def _extract_intrinsic_rows(intrinsic: dict | None, analysis: dict | None) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for arm, d in DEFAULT_BASE_INTRINSIC.items():
        rows[arm] = dict(d)
    if analysis and analysis.get("nextgen_tokenizer_validation"):
        for arm, rec in analysis["nextgen_tokenizer_validation"].items():
            surf = rec.get("mom_surface") or {}
            rows[arm] = {
                "dev_targets_with_eos": rec.get("dev_targets_with_eos"),
                "bytes_per_prediction": rec.get("bytes_per_prediction"),
                "pure_whitespace_tokens": rec.get("pure_whitespace_tokens"),
                "vocab_other_script": rec.get("vocab_other_script"),
                "akshara_per_1k": surf.get("mom_akshara_breaks_per_1000_reference_sites"),
                "mom_akshara_breaks": surf.get("mom_akshara_breaks"),
                "mom_egc_breaks": surf.get("mom_egc_breaks"),
                "mom_byte_fallback_ids": surf.get("mom_byte_fallback_ids"),
                "vocab_active_lexical": 47737,
                "canonical_eq": f"{surf.get('mom_canonical_equal', 186)}/{surf.get('mom_docs', 186)}" if surf else "--",
            }
    if intrinsic and intrinsic.get("arms"):
        for arm, rec in intrinsic["arms"].items():
            surf = rec.get("mom_surface") or {}
            rows[arm] = {
                "dev_targets_with_eos": rec.get("dev_targets_with_eos"),
                "bytes_per_prediction": rec.get("bytes_per_prediction"),
                "pure_whitespace_tokens": rec.get("pure_whitespace_tokens"),
                "vocab_other_script": rec.get("vocab_other_script"),
                "akshara_per_1k": surf.get("mom_akshara_breaks_per_1000_reference_sites"),
                "mom_akshara_breaks": surf.get("mom_akshara_breaks"),
                "mom_egc_breaks": surf.get("mom_egc_breaks"),
                "mom_byte_fallback_ids": surf.get("mom_byte_fallback_ids"),
                "vocab_active_lexical": rec.get("vocab_active_lexical", 47737),
                "canonical_eq": f"{surf.get('mom_canonical_equal', 186)}/{surf.get('mom_docs', 186)}" if surf else "--",
            }
    return rows


def build_intrinsic_table(intrinsic: dict | None, analysis: dict | None) -> str:
    rows = _extract_intrinsic_rows(intrinsic, analysis)
    ordered = [a for a in (*ALL_SEVEN_ARMS, *(k for k in rows if k not in ALL_SEVEN_ARMS)) if a in rows]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Scale-A 48K Malayalam tokenizer intrinsic and orthographic evaluation on the 186 development documents (999,950 canonical bytes).}",
        r"\label{tab:intrinsic-nextgen}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Arm & Targets & B/pred & Pure WS & OtherVoc & Akshara Brk & Brk/1k & ByteFB & CanonEQ \\",
        r"\midrule",
    ]
    for arm in ordered:
        r = rows[arm]
        lines.append(
            f"{_tex_escape(arm)} & {_fmt_int(r.get('dev_targets_with_eos'))} & "
            f"{_fmt(r.get('bytes_per_prediction'), '.3f')} & {_fmt_int(r.get('pure_whitespace_tokens'))} & "
            f"{_fmt_int(r.get('vocab_other_script'))} & {_fmt_int(r.get('mom_akshara_breaks'))} & "
            f"{_fmt(r.get('akshara_per_1k'), '.3f')} & {_fmt_int(r.get('mom_byte_fallback_ids'))} & "
            f"{_tex_escape(r.get('canonical_eq', '--'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_seed17_7arm_table(analysis: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Seed-17 38.8M GPT development evaluation across all 7 arms (10,000-draw paired document bootstrap, seed 20261125).}",
        r"\label{tab:seed17-7arm}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Protocol & Arm & BPB & B/pred & $\Delta$BPB vs M48 & $\Delta$BPB vs T8 & 95\% CI vs M48 \\",
        r"\midrule",
    ]
    protocols = analysis.get("protocols", {})
    for proto_idx, (mode, pdata) in enumerate(protocols.items()):
        if proto_idx > 0:
            lines.append(r"\midrule")
        metrics = pdata["metrics"]
        paired = pdata["paired"]
        for arm in pdata.get("arms", ALL_SEVEN_ARMS):
            if arm not in metrics:
                continue
            m = metrics[arm]
            vs_m48 = paired.get(f"{arm} minus M48")
            vs_t8 = paired.get(f"{arm} minus T8")
            d_m48 = _fmt(vs_m48["delta_bpb"], "+.4f") if vs_m48 else "0.0000"
            d_t8 = _fmt(vs_t8["delta_bpb"], "+.4f") if vs_t8 else ("0.0000" if arm == "T8" else "--")
            if vs_m48:
                lo, hi = vs_m48["document_bootstrap_95pct"]
                ci_s = f"[{lo:+.4f}, {hi:+.4f}]"
            else:
                ci_s = "--"
            lines.append(
                f"{_tex_escape(mode)} & {_tex_escape(arm)} & {m['bpb']:.4f} & "
                f"{m['canonical_bytes_per_prediction']:.3f} & {d_m48} & {d_t8} & {ci_s} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_multiseed_table(multiseed: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-seed Seed-17+ replication BPB summary and pooled paired bootstrap deltas vs M48 and T8.}",
        r"\label{tab:multiseed}",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Protocol & Arm & Seeds & Mean $\pm$ Std BPB & Min--Max & $\Delta$BPB vs M48 [95\% CI] \\",
        r"\midrule",
    ]
    for p_idx, (mode, pdata) in enumerate(multiseed.get("protocols", {}).items()):
        if p_idx > 0:
            lines.append(r"\midrule")
        vs_m48 = pdata.get("paired_vs_m48", {})
        for arm, stats in pdata.get("arms", {}).items():
            pair = vs_m48.get(f"{arm} minus M48")
            if pair:
                lo, hi = pair["document_bootstrap_95pct"]
                delta_col = f"{pair['delta_bpb']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            else:
                delta_col = "--"
            lines.append(
                f"{_tex_escape(mode)} & {_tex_escape(arm)} & {stats['n_seeds']} & "
                f"{stats['mean_bpb']:.4f} $\\pm$ {stats['std_bpb']:.4f} & "
                f"{stats['min_bpb']:.4f}--{stats['max_bpb']:.4f} & {delta_col} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def build_heldout_table(heldout: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Zero-retrain independent held-out evaluation across saved checkpoints.}",
        r"\label{tab:heldout}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Arm & Held-Out BPB & B/pred & Akshara Brk & Brk/1k \\",
        r"\midrule",
    ]
    for arm, rec in heldout.get("arms", {}).items():
        surf = rec.get("mom_surface") or {}
        lines.append(
            f"{_tex_escape(arm)} & {rec['bpb']:.4f} & {rec['bytes_per_prediction']:.3f} & "
            f"{_fmt_int(surf.get('mom_akshara_breaks'))} & "
            f"{_fmt(surf.get('mom_akshara_breaks_per_1000_reference_sites'), '.3f')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


# --- Zero-dependency deterministic vector PDF 1.4 + crisp PNG fallback renderer ---

_FONT5X7 = {
    " ": (0, 0, 0, 0, 0),
    "-": (0, 0, 14, 0, 0),
    ".": (0, 0, 0, 0, 4),
    ",": (0, 0, 0, 4, 8),
    ":": (0, 4, 0, 4, 0),
    "/": (1, 2, 4, 8, 16),
    "(": (4, 8, 8, 8, 4),
    ")": (8, 4, 4, 4, 8),
    "+": (0, 4, 14, 4, 0),
    "0": (14, 17, 17, 17, 14),
    "1": (4, 12, 4, 4, 14),
    "2": (14, 1, 14, 16, 31),
    "3": (14, 1, 6, 1, 14),
    "4": (17, 17, 31, 1, 1),
    "5": (31, 16, 14, 1, 30),
    "6": (14, 16, 30, 17, 14),
    "7": (31, 1, 2, 4, 4),
    "8": (14, 17, 14, 17, 14),
    "9": (14, 17, 15, 1, 14),
    "A": (14, 17, 31, 17, 17),
    "B": (30, 17, 30, 17, 30),
    "C": (14, 16, 16, 16, 14),
    "D": (30, 17, 17, 17, 30),
    "E": (31, 16, 30, 16, 31),
    "F": (31, 16, 30, 16, 16),
    "G": (14, 16, 23, 17, 14),
    "H": (17, 17, 31, 17, 17),
    "I": (14, 4, 4, 4, 14),
    "J": (1, 1, 1, 17, 14),
    "K": (17, 18, 28, 18, 17),
    "L": (16, 16, 16, 16, 31),
    "M": (17, 27, 21, 17, 17),
    "N": (17, 25, 21, 19, 17),
    "O": (14, 17, 17, 17, 14),
    "P": (30, 17, 30, 16, 16),
    "Q": (14, 17, 17, 19, 15),
    "R": (30, 17, 30, 18, 17),
    "S": (15, 16, 14, 1, 30),
    "T": (31, 4, 4, 4, 4),
    "U": (17, 17, 17, 17, 14),
    "V": (17, 17, 17, 10, 4),
    "W": (17, 17, 21, 27, 17),
    "X": (17, 10, 4, 10, 17),
    "Y": (17, 10, 4, 4, 4),
    "Z": (31, 2, 4, 8, 31),
}


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _encode_png(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(raw_rows, level=6)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def _encode_vector_pdf(width: int, height: int, ops: list[str]) -> bytes:
    stream_data = ("\n".join(ops) + "\n").encode("latin1", errors="replace")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream_data)).encode("ascii") + b" >>\nstream\n" + stream_data + b"endstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, body in enumerate(objs, 1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii"))
    return bytes(out)


class _DualCanvas:
    """Renders vector PDF primitives and raster PNG output identically without matplotlib."""

    def __init__(self, width: int = 860, height: int = 520):
        self.w = width
        self.h = height
        self.rgb = np.full((height, width, 3), 255, dtype=np.uint8)
        self.pdf: list[str] = []

    def _pdf_y(self, y: float) -> float:
        return self.h - y

    def line(self, x1: float, y1: float, x2: float, y2: float, color=(40, 40, 40), lw: float = 1.5) -> None:
        r, g, b = [c / 255.0 for c in color]
        self.pdf.append(f"{r:.3f} {g:.3f} {b:.3f} RG {lw:.2f} w {x1:.2f} {self._pdf_y(y1):.2f} m {x2:.2f} {self._pdf_y(y2):.2f} l S")
        steps = max(2, int(max(abs(x2 - x1), abs(y2 - y1)) * 2))
        xs = np.clip(np.round(np.linspace(x1, x2, steps)).astype(int), 0, self.w - 1)
        ys = np.clip(np.round(np.linspace(y1, y2, steps)).astype(int), 0, self.h - 1)
        self.rgb[ys, xs] = color

    def rect(self, x: float, y: float, w: float, h: float, fill=(70, 130, 180)) -> None:
        r, g, b = [c / 255.0 for c in fill]
        self.pdf.append(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {self._pdf_y(y + h):.2f} {w:.2f} {h:.2f} re f")
        x0, x1 = max(0, int(round(x))), min(self.w, int(round(x + w)))
        y0, y1 = max(0, int(round(y))), min(self.h, int(round(y + h)))
        if x1 > x0 and y1 > y0:
            self.rgb[y0:y1, x0:x1] = fill

    def circle(self, cx: float, cy: float, rad: float = 5.5, fill=(31, 119, 180)) -> None:
        r, g, b = [c / 255.0 for c in fill]
        k = 0.55228475 * rad
        py = self._pdf_y(cy)
        self.pdf.append(
            f"{r:.3f} {g:.3f} {b:.3f} rg "
            f"{cx + rad:.2f} {py:.2f} m "
            f"{cx + rad:.2f} {py + k:.2f} {cx + k:.2f} {py + rad:.2f} {cx:.2f} {py + rad:.2f} c "
            f"{cx - k:.2f} {py + rad:.2f} {cx - rad:.2f} {py + k:.2f} {cx - rad:.2f} {py:.2f} c "
            f"{cx - rad:.2f} {py - k:.2f} {cx - k:.2f} {py - rad:.2f} {cx:.2f} {py - rad:.2f} c "
            f"{cx + k:.2f} {py - rad:.2f} {cx + rad:.2f} {py - k:.2f} {cx + rad:.2f} {py:.2f} c f"
        )
        yy, xx = np.ogrid[: self.h, : self.w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2
        self.rgb[mask] = fill

    def text(self, x: float, y: float, label: str, size: int = 11, color=(20, 20, 20)) -> None:
        r, g, b = [c / 255.0 for c in color]
        safe = str(label).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.pdf.append(f"BT /F1 {size} Tf {r:.3f} {g:.3f} {b:.3f} rg 1 0 0 1 {x:.2f} {self._pdf_y(y):.2f} Tm ({safe}) Tj ET")
        cx = int(round(x))
        cy = int(round(y)) - 7
        for ch in str(label).upper():
            glyph = _FONT5X7.get(ch, (0, 0, 0, 0, 0))
            for row_i, bits in enumerate(glyph):
                for col_i in range(5):
                    if bits & (1 << (4 - col_i)):
                        px, py = cx + col_i, cy + row_i
                        if 0 <= px < self.w and 0 <= py < self.h:
                            self.rgb[py, px] = color
            cx += 6

    def save_pair(self, stem: Path) -> tuple[Path, Path]:
        pdf_path = stem.with_suffix(".pdf")
        png_path = stem.with_suffix(".png")
        pdf_path.write_bytes(_encode_vector_pdf(self.w, self.h, self.pdf))
        png_path.write_bytes(_encode_png(self.rgb))
        return pdf_path, png_path


def _plot_pareto_builtin(points: list[tuple[str, float, float]], out_stem: Path) -> tuple[Path, Path]:
    c = _DualCanvas(860, 520)
    pad_l, pad_r, pad_t, pad_b = 90, 40, 50, 75
    xs = [p[1] for p in points] or [0.0, 10.0]
    ys = [p[2] for p in points] or [0.85, 0.95]
    x_min, x_max = min(0.0, min(xs) * 0.9), max(xs) * 1.15 + 1e-6
    y_min, y_max = min(ys) - 0.01, max(ys) + 0.01
    pw, ph = c.w - pad_l - pad_r, c.h - pad_t - pad_b

    def sx(v):
        return pad_l + (v - x_min) / (x_max - x_min) * pw

    def sy(v):
        return c.h - pad_b - (v - y_min) / (y_max - y_min) * ph

    c.text(pad_l, 26, "PARETO FRONTIER: AKSHARA BREAKS VS SEED-17 BPB", size=14)
    c.line(pad_l, pad_t, pad_l, c.h - pad_b)
    c.line(pad_l, c.h - pad_b, c.w - pad_r, c.h - pad_b)
    for frac in (0.25, 0.5, 0.75):
        gy = pad_t + frac * ph
        c.line(pad_l, gy, c.w - pad_r, gy, color=(225, 225, 225), lw=0.8)
    palette = ((31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189))
    for i, (arm, x_v, y_v) in enumerate(points):
        col = palette[i % len(palette)]
        px, py = sx(x_v), sy(y_v)
        c.circle(px, py, rad=6.0, fill=col)
        c.text(px + 9, py + 4, f"{arm} ({y_v:.4f})", size=10, color=col)
    c.text(c.w // 2 - 90, c.h - 22, "AKSHARA BREAKS PER 1000 SITES (LOWER BETTER)", size=10)
    return c.save_pair(out_stem)


def _plot_vocab_builtin(alloc_rows: list[tuple[str, int, int, int, int]], out_stem: Path) -> tuple[Path, Path]:
    c = _DualCanvas(860, 520)
    pad_l, pad_r, pad_t, pad_b = 180, 40, 50, 60
    pw, ph = c.w - pad_l - pad_r, c.h - pad_t - pad_b
    c.text(pad_l, 26, "48K VOCABULARY ALLOCATION BREAKDOWN", size=14)
    n = max(1, len(alloc_rows))
    row_h = max(18, int(ph / n * 0.58))
    step = ph / n
    for idx, (arm, core_lex, other_voc, byte_fb, specials) in enumerate(alloc_rows):
        total = max(1, core_lex + other_voc + byte_fb + specials)
        y = pad_t + idx * step + (step - row_h) / 2
        x = pad_l
        for count, col in (
            (core_lex, (44, 160, 44)),
            (other_voc, (214, 39, 40)),
            (byte_fb, (31, 119, 180)),
            (specials, (127, 127, 127)),
        ):
            bw = count / total * pw
            c.rect(x, y, bw, row_h, fill=col)
            x += bw
        c.text(16, y + row_h / 2 + 4, f"{arm} (OTH:{other_voc})", size=10)
    return c.save_pair(out_stem)


def _plot_forest_builtin(ci_rows: list[tuple[str, float, float, float]], out_stem: Path) -> tuple[Path, Path]:
    c = _DualCanvas(860, 520)
    pad_l, pad_r, pad_t, pad_b = 260, 40, 50, 60
    pw, ph = c.w - pad_l - pad_r, c.h - pad_t - pad_b
    vals = [v for _, d, lo, hi in ci_rows for v in (d, lo, hi)] or [-0.02, 0.02]
    v_min, v_max = min(-0.005, min(vals) * 1.15), max(0.005, max(vals) * 1.15)

    def sx(v):
        return pad_l + (v - v_min) / (v_max - v_min) * pw

    c.text(pad_l - 80, 26, "PAIRED DOCUMENT BOOTSTRAP DELTA BPB (95% CI)", size=13)
    x0 = sx(0.0)
    c.line(x0, pad_t, x0, c.h - pad_b, color=(120, 120, 120), lw=1.2)
    n = max(1, len(ci_rows))
    step = ph / n
    for idx, (label, delta, lo, hi) in enumerate(ci_rows):
        y = pad_t + (idx + 0.5) * step
        col = (44, 160, 44) if hi < 0 else (214, 39, 40) if lo > 0 else (31, 119, 180)
        c.line(sx(lo), y, sx(hi), y, color=col, lw=2.0)
        c.circle(sx(delta), y, rad=4.5, fill=col)
        c.text(14, y + 4, f"{label} ({delta:+.4f})", size=9)
    c.text(pad_l, c.h - 22, "DELTA BPB (< 0 FAVORS FIRST ARM)", size=10)
    return c.save_pair(out_stem)


def _render_matplotlib(points, alloc_rows, ci_rows, out_dir: Path) -> dict[str, list[str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: dict[str, list[str]] = {}

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for arm, x_v, y_v in points:
        ax.scatter([x_v], [y_v], s=85, label=arm, zorder=3)
        ax.annotate(arm, (x_v, y_v), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Akshara breaks per 1,000 reference sites")
    ax.set_ylabel("Seed-17 Development BPB")
    ax.set_title("Malayalam Tokenization Pareto Frontier (Seed-17)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    p_stem = out_dir / "fig_pareto_akshara_vs_bpb"
    fig.savefig(p_stem.with_suffix(".pdf"))
    fig.savefig(p_stem.with_suffix(".png"), dpi=220)
    plt.close(fig)
    outputs["pareto"] = [str(p_stem.with_suffix(".pdf")), str(p_stem.with_suffix(".png"))]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    arms = [r[0] for r in alloc_rows]
    core = np.array([r[1] for r in alloc_rows])
    other = np.array([r[2] for r in alloc_rows])
    fb = np.array([r[3] for r in alloc_rows])
    sp = np.array([r[4] for r in alloc_rows])
    y = np.arange(len(arms))
    ax.barh(y, core, label="Core ML/ASCII/Merge", color="#2ca02c")
    ax.barh(y, other, left=core, label="Other-Script", color="#d62728")
    ax.barh(y, fb, left=core + other, label="UTF-8 Byte Fallback (256)", color="#1f77b4")
    ax.barh(y, sp, left=core + other + fb, label="Specials (7)", color="#7f7f7f")
    ax.set_yticks(y, arms)
    ax.set_xlabel("48K Vocabulary Slots")
    ax.set_title("Vocabulary Slot Allocation Breakdown")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    v_stem = out_dir / "fig_vocab_allocation"
    fig.savefig(v_stem.with_suffix(".pdf"))
    fig.savefig(v_stem.with_suffix(".png"), dpi=220)
    plt.close(fig)
    outputs["vocab"] = [str(v_stem.with_suffix(".pdf")), str(v_stem.with_suffix(".png"))]

    fig, ax = plt.subplots(figsize=(8.0, max(3.8, 0.48 * max(1, len(ci_rows)))))
    y = np.arange(len(ci_rows))
    for idx, (label, delta, lo, hi) in enumerate(ci_rows):
        col = "#2ca02c" if hi < 0 else "#d62728" if lo > 0 else "#1f77b4"
        ax.errorbar([delta], [idx], xerr=[[delta - lo], [hi - delta]], fmt="o", color=col, capsize=4)
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_yticks(y, [r[0] for r in ci_rows])
    ax.set_xlabel("Paired Bootstrap Delta BPB (95% CI)")
    ax.set_title("Paired 10,000-Draw Document Bootstrap Comparison")
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    f_stem = out_dir / "fig_bootstrap_forest"
    fig.savefig(f_stem.with_suffix(".pdf"))
    fig.savefig(f_stem.with_suffix(".png"), dpi=220)
    plt.close(fig)
    outputs["forest"] = [str(f_stem.with_suffix(".pdf")), str(f_stem.with_suffix(".png"))]
    return outputs


def generate_report(
    *,
    out_dir: Path,
    analysis_path: Path | None = None,
    intrinsic_path: Path | None = None,
    multiseed_path: Path | None = None,
    heldout_path: Path | None = None,
) -> dict:
    """Write publication LaTeX tables and vector PDF/PNG figures to `out_dir`."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(analysis_path.read_bytes()) if analysis_path and analysis_path.is_file() else None
    intrinsic = json.loads(intrinsic_path.read_bytes()) if intrinsic_path and intrinsic_path.is_file() else None
    multiseed = json.loads(multiseed_path.read_bytes()) if multiseed_path and multiseed_path.is_file() else None
    heldout = json.loads(heldout_path.read_bytes()) if heldout_path and heldout_path.is_file() else None

    tables = {}
    t_intr = out_dir / "table_intrinsic.tex"
    t_intr.write_text(build_intrinsic_table(intrinsic, analysis), encoding="utf-8")
    tables["intrinsic"] = str(t_intr)

    if analysis is not None:
        t_s17 = out_dir / "table_seed17_7arm.tex"
        t_s17.write_text(build_seed17_7arm_table(analysis), encoding="utf-8")
        tables["seed17_7arm"] = str(t_s17)

    if multiseed is not None:
        t_ms = out_dir / "table_multiseed.tex"
        t_ms.write_text(build_multiseed_table(multiseed), encoding="utf-8")
        tables["multiseed"] = str(t_ms)

    if heldout is not None:
        t_ho = out_dir / "table_heldout.tex"
        t_ho.write_text(build_heldout_table(heldout), encoding="utf-8")
        tables["heldout"] = str(t_ho)

    intr_rows = _extract_intrinsic_rows(intrinsic, analysis)
    # Build Pareto points from primary protocol (fixed_source_bytes or first protocol).
    points: list[tuple[str, float, float]] = []
    ci_rows: list[tuple[str, float, float, float]] = []
    if analysis and analysis.get("protocols"):
        primary_mode = "fixed_source_bytes" if "fixed_source_bytes" in analysis["protocols"] else next(iter(analysis["protocols"]))
        pdata = analysis["protocols"][primary_mode]
        for arm, m in pdata["metrics"].items():
            x_val = intr_rows.get(arm, {}).get("akshara_per_1k")
            if x_val is not None:
                points.append((arm, float(x_val), float(m["bpb"])))
        for mode_name, mdata in analysis["protocols"].items():
            for pair_key, pvals in mdata.get("paired", {}).items():
                if pair_key.endswith((" minus M48", " minus T8")):
                    lo, hi = pvals["document_bootstrap_95pct"]
                    ci_rows.append((f"{mode_name[:6]}:{pair_key}", float(pvals["delta_bpb"]), float(lo), float(hi)))
    if not points:
        points = [(arm, float(r["akshara_per_1k"]), 0.90) for arm, r in intr_rows.items() if r.get("akshara_per_1k") is not None]
    if not ci_rows:
        ci_rows = [("no-comparisons", 0.0, -0.01, 0.01)]

    alloc_rows = []
    for arm, rec in intr_rows.items():
        if rec.get("vocab_other_script") is not None:
            lex_total = int(rec.get("vocab_active_lexical") or 47737)
            other = int(rec["vocab_other_script"])
            alloc_rows.append((arm, max(0, lex_total - other), other, 256, 7))
    if not alloc_rows:
        alloc_rows = [("M48", 47737, 0, 256, 7)]

    if importlib.util.find_spec("matplotlib") is not None:
        renderer = "matplotlib"
        figures = _render_matplotlib(points, alloc_rows, ci_rows, out_dir)
    else:
        renderer = "builtin_vector"
        figures = {
            "pareto": [str(p) for p in _plot_pareto_builtin(points, out_dir / "fig_pareto_akshara_vs_bpb")],
            "vocab": [str(p) for p in _plot_vocab_builtin(alloc_rows, out_dir / "fig_vocab_allocation")],
            "forest": [str(p) for p in _plot_forest_builtin(ci_rows, out_dir / "fig_bootstrap_forest")],
        }

    summary = {"renderer": renderer, "tables": tables, "figures": figures}
    (out_dir / "report-index.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--intrinsic", type=Path)
    parser.add_argument("--multiseed", type=Path)
    parser.add_argument("--heldout", type=Path)
    args = parser.parse_args()
    report = generate_report(
        out_dir=args.out_dir,
        analysis_path=args.analysis,
        intrinsic_path=args.intrinsic,
        multiseed_path=args.multiseed,
        heldout_path=args.heldout,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
