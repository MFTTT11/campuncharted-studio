"""High-CTR Pillow thumbnail generator with smart visual-energy frame selection."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat

from .audio_engine import probe_audio_duration
from .subtitle_engine import detect_bold_font_path


def _score_frame(image_path: Path) -> float:
    """Score frame visual energy using sharpness, saturation, contrast, and balanced brightness."""
    with Image.open(image_path) as raw:
        rgb = raw.convert("RGB").resize((320, 180), Image.Resampling.BILINEAR)
        hsv = rgb.convert("HSV")
        gray = rgb.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)

        rgb_stat = ImageStat.Stat(rgb)
        hsv_stat = ImageStat.Stat(hsv)
        edge_stat = ImageStat.Stat(edges)

        brightness = sum(rgb_stat.mean) / 3.0
        contrast = sum(rgb_stat.stddev) / 3.0
        saturation = hsv_stat.mean[1]
        sharpness = edge_stat.mean[0]

        # Penalize nearly black or overexposed white frames
        brightness_penalty = 0.0
        if brightness < 35:
            brightness_penalty = (35 - brightness) * 1.5
        elif brightness > 220:
            brightness_penalty = (brightness - 220) * 1.5

        return (sharpness * 1.6) + (saturation * 0.55) + (contrast * 0.85) - brightness_penalty


def extract_best_frame(mp4_path: Path, work_dir: Path, *, sample_count: int = 7) -> Path:
    """Extract evenly spaced frames from MP4 and return the highest visual-energy frame."""
    duration = probe_audio_duration(mp4_path)
    frames_dir = work_dir / "thumb_candidates"
    frames_dir.mkdir(parents=True, exist_ok=True)

    best_path: Path | None = None
    best_score = float("-inf")
    for idx in range(sample_count):
        frac = 0.12 + (0.72 * idx / max(1, sample_count - 1))
        ts = max(0.3, min(duration - 0.3, duration * frac))
        out_png = frames_dir / f"cand_{idx:02d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(mp4_path),
                "-frames:v",
                "1",
                str(out_png),
            ],
            check=True,
            timeout=30,
        )
        if out_png.is_file():
            score = _score_frame(out_png)
            if score > best_score:
                best_score = score
                best_path = out_png

    if not best_path:
        raise RuntimeError(f"Failed to extract thumbnail candidate frames from {mp4_path}")
    return best_path


def _load_truetype(size: int) -> ImageFont.FreeTypeFont:
    font_path = detect_bold_font_path()
    if font_path is None:
        raise RuntimeError("No bold TrueType font installed (install fonts-roboto or fonts-dejavu-core).")
    return ImageFont.truetype(str(font_path), size=size)


def _fit_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    scale = max(target_w / max(1, src_w), target_h / max(1, src_h))
    new_w = max(target_w, int(round(src_w * scale)))
    new_h = max(target_h, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def _split_hook_lines(hook_text: str) -> list[str]:
    words = [w.strip() for w in hook_text.upper().split() if w.strip()]
    if not words:
        return ["WILDERNESS", "TRUTH"]
    if len(words) <= 2:
        return words
    mid = (len(words) + 1) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def generate_thumbnail(
    source_frame: Path,
    out_jpg: Path,
    *,
    hook_text: str,
    sub_label: str = "",
    format_type: str = "documentary",
) -> Path:
    """Render a crisp, bold, high-CTR JPEG thumbnail (<1.9 MB) for Documentary (16:9) or Short (9:16)."""
    out_jpg.parent.mkdir(parents=True, exist_ok=True)
    is_short = format_type == "short"
    width, height = (1080, 1920) if is_short else (1280, 720)

    with Image.open(source_frame) as raw:
        canvas = _fit_cover(raw.convert("RGB"), width, height)

    # Punchy cinematic grading so the cover stands out on mobile feeds
    canvas = ImageEnhance.Color(canvas).enhance(1.18)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.12)
    canvas = ImageEnhance.Sharpness(canvas).enhance(1.18)

    # Smooth dark gradient scrim for crisp contrast behind typography
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ov_px = overlay.load()
    grad_start = int(height * (0.48 if is_short else 0.36))
    for y in range(height):
        alpha = 0
        if y < int(height * 0.16):
            frac = 1.0 - (y / float(height * 0.16))
            alpha = int(120 * (frac**1.5))
        if y >= grad_start:
            frac = (y - grad_start) / float(max(1, height - grad_start))
            alpha = max(alpha, int(210 * (frac**1.35)))
        if alpha > 0:
            for x in range(width):
                ov_px[x, y] = (7, 10, 15, alpha)

    composed = Image.alpha_composite(canvas.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(composed)

    gold = (255, 212, 59, 255)
    white = (255, 255, 255, 255)
    dark_stroke = (10, 12, 16, 255)
    pill_bg = (12, 16, 22, 205)

    # Top brand/topic badge pill
    badge_text = (sub_label.strip() or "CAMPUNCHARTED").upper()[:28]
    badge_font = _load_truetype(34 if is_short else 26)
    bbox = draw.textbbox((0, 0), badge_text, font=badge_font)
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    pad_x, pad_y = (28, 14) if is_short else (22, 11)
    if is_short:
        bx0 = (width - (bw + 2 * pad_x)) // 2
        by0 = 135
    else:
        bx0 = 64
        by0 = 48
    draw.rounded_rectangle(
        [bx0, by0, bx0 + bw + 2 * pad_x, by0 + bh + 2 * pad_y],
        radius=16,
        fill=pill_bg,
        outline=gold,
        width=3,
    )
    draw.text((bx0 + pad_x, by0 + pad_y - 2), badge_text, font=badge_font, fill=gold)

    # Dynamic huge impact hook typography (auto-fits width cleanly)
    lines = _split_hook_lines(hook_text)
    max_text_w = int(width * (0.86 if is_short else 0.84))
    font_size = 138 if is_short else 118
    font = _load_truetype(font_size)
    while font_size > 54:
        font = _load_truetype(font_size)
        widest = max((draw.textbbox((0, 0), line, font=font)[2] for line in lines), default=0)
        if widest <= max_text_w:
            break
        font_size -= 6

    line_gap = int(font_size * 0.14)
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total_h = sum(line_heights) + line_gap * max(0, len(lines) - 1)
    stroke_w = max(5, font_size // 14)

    if is_short:
        start_y = int(height * 0.68) - total_h // 2
        for idx, line in enumerate(lines):
            lb = draw.textbbox((0, 0), line, font=font)
            lw = lb[2] - lb[0]
            tx = (width - lw) // 2
            color = gold if idx == len(lines) - 1 else white
            draw.text((tx + 5, start_y + 7), line, font=font, fill=(0, 0, 0, 175))
            draw.text(
                (tx, start_y),
                line,
                font=font,
                fill=color,
                stroke_width=stroke_w,
                stroke_fill=dark_stroke,
            )
            start_y += line_heights[idx] + line_gap
    else:
        # 16:9 Documentary cover: bold lower-left editorial layout with gold accent bar
        bar_x = 64
        start_y = height - total_h - 68
        draw.rounded_rectangle([bar_x, start_y + 8, bar_x + 14, start_y + total_h - 4], radius=6, fill=gold)
        tx = bar_x + 36
        for idx, line in enumerate(lines):
            color = gold if idx == len(lines) - 1 else white
            draw.text((tx + 5, start_y + 6), line, font=font, fill=(0, 0, 0, 185))
            draw.text(
                (tx, start_y),
                line,
                font=font,
                fill=color,
                stroke_width=stroke_w,
                stroke_fill=dark_stroke,
            )
            start_y += line_heights[idx] + line_gap

    # Save under YouTube's strict 2 MB custom thumbnail limit
    final_rgb = composed.convert("RGB")
    for quality in (92, 86, 78):
        final_rgb.save(out_jpg, "JPEG", quality=quality, optimize=True, progressive=True)
        if out_jpg.stat().st_size <= 1_900_000:
            break
    return out_jpg
