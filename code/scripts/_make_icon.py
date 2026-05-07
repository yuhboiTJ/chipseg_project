"""one-off: generate code/icon.png (64x64) and code/icon.ico (multi-size)."""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def make_icon(size=64):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # silicon chip body: dark gray rounded square with a thin border
    pad = max(2, size // 10)
    radius = max(2, size // 10)
    chip_box = (pad, pad, size - pad - 1, size - pad - 1)
    d.rounded_rectangle(chip_box, radius=radius,
                        fill=(55, 60, 65, 255),
                        outline=(150, 155, 160, 255),
                        width=max(1, size // 64))

    # seedable area: slightly inset green fill so the chip looks like the
    # live overlay (chip with a green seedable region inside).
    inner_pad = pad + max(2, size // 12)
    inner_box = (inner_pad, inner_pad, size - inner_pad - 1, size - inner_pad - 1)
    d.rounded_rectangle(inner_box, radius=max(2, radius - 2),
                        fill=(40, 150, 80, 220))

    # crosshair: yellow lines with a gap in the middle, matching live_demo
    cx, cy = size // 2, size // 2
    arm = size // 3
    gap = max(2, size // 12)
    yellow = (255, 220, 60, 255)
    line_w = max(2, size // 32)
    d.line([(cx - arm, cy), (cx - gap, cy)], fill=yellow, width=line_w)
    d.line([(cx + gap, cy), (cx + arm, cy)], fill=yellow, width=line_w)
    d.line([(cx, cy - arm), (cx, cy - gap)], fill=yellow, width=line_w)
    d.line([(cx, cy + gap), (cx, cy + arm)], fill=yellow, width=line_w)
    r = max(2, size // 24)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=yellow, width=line_w)

    return img


def main():
    png_path = ROOT / "icon.png"
    ico_path = ROOT / "icon.ico"

    icon_64 = make_icon(64)
    icon_64.save(png_path, format="PNG")
    print(f"wrote icon.png  ({png_path.stat().st_size} bytes)")

    icon_big = make_icon(256)
    icon_big.save(ico_path, format="ICO",
                  sizes=[(16, 16), (32, 32), (48, 48), (64, 64),
                         (128, 128), (256, 256)])
    print(f"wrote icon.ico  ({ico_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
