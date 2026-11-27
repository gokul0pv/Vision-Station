"""Draws ui/app.ico, the Vision Core icon: a lens aperture in the Studio's
accent gradient on a rounded dark square. Run once; the .ico is committed."""
import math
import os

from PIL import Image, ImageDraw

S = 1024
BG, A1, A2 = (11, 13, 18), (124, 92, 255), (79, 140, 255)


def lerp(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([40, 40, S - 40, S - 40], radius=220, fill=BG + (255,))
# accent ring, gradient top-left to bottom-right
ring = Image.new("RGBA", (S, S), (0, 0, 0, 0))
g = Image.new("RGBA", (S, S))
gp = g.load()
for y in range(S):
    for x in range(S):
        gp[x, y] = lerp(A1, A2, (x + y) / (2.0 * S)) + (255,)
mask = Image.new("L", (S, S), 0)
md = ImageDraw.Draw(mask)
c, R, r = S / 2, 330, 250
md.ellipse([c - R, c - R, c + R, c + R], fill=255)
md.ellipse([c - r, c - r, c + r, c + r], fill=0)
# aperture blades: six wedges inside the ring
for k in range(6):
    a0 = math.radians(k * 60 + 15)
    pts = [(c + r * math.cos(a0), c + r * math.sin(a0)),
           (c + r * math.cos(a0 + math.radians(60)), c + r * math.sin(a0 + math.radians(60))),
           (c + 95 * math.cos(a0 + math.radians(100)), c + 95 * math.sin(a0 + math.radians(100)))]
    md.polygon(pts, fill=255 if k % 2 == 0 else 170)
md.ellipse([c - 88, c - 88, c + 88, c + 88], fill=0)
img.paste(g, (0, 0), mask)
img.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.ico"),
         sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

print("ui/app.ico written")
