#!/usr/bin/env python3
"""Generate the zonos3-app icon set (assets/app/zonos3.{png,ico,icns}).

Draws a 1024x1024 mark in the style of assets/ZONOS3BlogThumbnail.png (near-black
field, orange smoke wave, thin silver "Z3") — the thumbnail itself is 1200x628
wordmark art and doesn't crop to a square. Pillow renders all three outputs, so
this runs on any platform (no iconutil/ImageMagick needed).

    python3 scripts/make-app-icon.py [outdir=assets/app]
"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE = 1024
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/lato/Lato-Light.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "C:/Windows/Fonts/segoeuil.ttf",
]


def load_font(px: int) -> ImageFont.FreeTypeFont:
    for cand in FONT_CANDIDATES:
        if Path(cand).exists():
            return ImageFont.truetype(cand, px)
    return ImageFont.load_default(px)


def wave_layer(color, amp, phase, thickness, blur, alpha):
    """One blurred sine ribbon across the lower half, like the thumbnail's smoke."""
    layer = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pts = []
    for x in range(-64, SIZE + 65, 16):
        y = SIZE * 0.68 + amp * math.sin(x / SIZE * 2.2 * math.pi + phase)
        pts.append((x, y))
    for off in range(-thickness // 2, thickness // 2 + 1, 6):
        d.line([(x, y + off) for x, y in pts], fill=color + (alpha,), width=8)
    return layer.filter(ImageFilter.GaussianBlur(blur))


def make_master(phase: float = 0.0) -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (10, 11, 14, 255))

    # subtle radial glow behind the mark
    glow = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(glow).ellipse((112, 112, SIZE - 112, SIZE - 112), fill=46)
    img.paste(Image.new("RGBA", (SIZE, SIZE), (46, 42, 36, 255)), (0, 0),
              glow.filter(ImageFilter.GaussianBlur(180)))

    # orange smoke waves (front) + a faint silver one (back)
    img.alpha_composite(wave_layer((190, 190, 200), 120, 2.4 + phase, 70, 60, 60))
    img.alpha_composite(wave_layer((222, 130, 28), 96, phase, 90, 40, 130))
    img.alpha_composite(wave_layer((255, 170, 60), 110, 0.9 + phase, 40, 18, 160))

    # thin silver "Z3", slightly letterspaced, optically centered in the upper field
    font = load_font(430)
    text = "Z3"
    tracking = 40
    widths = [ImageDraw.Draw(img).textlength(c, font=font) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = (SIZE - total) / 2
    top = SIZE * 0.42
    grad = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for i, c in enumerate(text):
        gd.text((x, top), c, font=font, fill=(233, 235, 240, 255), anchor="lm")
        x += widths[i] + tracking
    img.alpha_composite(grad)
    return img


def main() -> None:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("assets/app")
    outdir.mkdir(parents=True, exist_ok=True)
    master = make_master()

    master.save(outdir / "zonos3.png")
    master.save(outdir / "zonos3.ico", sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    # Pillow's ICNS writer derives all required sizes from the 1024px master.
    master.save(outdir / "zonos3.icns")
    # Regenerate project artwork as Z3, not renamed pixels with legacy branding.
    assets = outdir.parent
    def artwork(phase=0.0):
        image = Image.new("RGB", (1200, 628), (10, 11, 14))
        mark = make_master(phase).convert("RGB").resize((600, 600), Image.Resampling.LANCZOS)
        image.paste(mark, (18, 14))
        draw = ImageDraw.Draw(image)
        draw.text((638, 244), "Zonos3", font=load_font(106), fill=(233, 235, 240))
        draw.text((642, 380), "Native speech synthesis", font=load_font(33), fill=(190, 190, 200))
        draw.text((642, 438), "zonos3.cpp", font=load_font(36), fill=(255, 170, 60))
        return image
    artwork().save(assets / "ZONOS3BlogThumbnail.png")
    frames = [artwork(i * 2 * math.pi / 24).resize((600, 314), Image.Resampling.LANCZOS)
              for i in range(24)]
    frames[0].save(assets / "zonos3_arlooop_animated.gif", save_all=True,
                   append_images=frames[1:], duration=90, loop=0, disposal=2)
    print(f"wrote {outdir}/zonos3.png .ico .icns and Zonos3 artwork")


if __name__ == "__main__":
    main()
