"""Contact sheets for the user's look (M4 step 6c).

    uv run python dev/research/m4_selection/sheets.py

first50_<method>.jpg: each method's first 50 picks at seed 0 (rank order, 10 per row).
quiet_sample.jpg: 12 quiet candidates (17 h) with SAM3's boxes >= 0.5 and their scores.
"""
from pathlib import Path

from PIL import Image, ImageDraw

from rin import Master

HERE = Path(__file__).parent
CANDS = "20261127-210554-curate.dedupe-6a1d"
LABELS = ["20261127-111139-sam3.concept-5477", "20261127-210937-sam3.concept-83bc"]
FIRST = {"random": "20261127-211121-curate.select-8f6c", "kcenter": "20261127-211121-curate.select-49c7",
         "kmeans_stratified": "20261127-211148-curate.select-f500"}

m = Master("workspace/store")


def image(media_id: str, idx: int) -> Image.Image:
    return Image.open(m.layout.root / "frames" / media_id / f"{idx}.jpg").convert("RGB")


for method, run_id in FIRST.items():
    rows = m.sql("SELECT media_id, frame_idx FROM frame_sets WHERE run_id = ? ORDER BY rank LIMIT 50", [run_id]).fetchall()
    w, h = 256, 144
    sheet = Image.new("RGB", (10 * w, 5 * h))
    for k, (media_id, idx) in enumerate(rows):
        sheet.paste(image(media_id, idx).resize((w, h)), ((k % 10) * w, (k // 10) * h))
    sheet.save(HERE / f"first50_{method}.jpg", quality=85)

quiet = m.sql("""SELECT s.media_id, s.frame_idx, d.camera FROM frame_sets s JOIN media d USING (media_id)
                 WHERE s.run_id = ? AND hour(d.started_at) >= 17 ORDER BY hash(s.media_id || s.frame_idx) LIMIT 12""",
              [CANDS]).fetchall()
w, h = 640, 360
sheet = Image.new("RGB", (2 * w, 6 * h))
for k, (media_id, idx, camera) in enumerate(quiet):
    im = image(media_id, idx)
    draw = ImageDraw.Draw(im)
    for x0, y0, x1, y1, score in m.sql(
        """SELECT x0, y0, x1, y1, score FROM instances WHERE list_contains(?, run_id)
           AND media_id = ? AND frame_idx = ? AND score >= 0.5""", [LABELS, media_id, idx]).fetchall():
        draw.rectangle([x0 * im.width, y0 * im.height, x1 * im.width, y1 * im.height], outline="red", width=4)
        draw.text((x0 * im.width, y0 * im.height - 22), f"{score:.2f}", fill="yellow", font_size=22)
    im = im.resize((w, h))
    ImageDraw.Draw(im).text((6, 6), f"{camera} frame {idx}", fill="cyan", font_size=18)
    sheet.paste(im, ((k % 2) * w, (k // 2) * h))
sheet.save(HERE / "quiet_sample.jpg", quality=85)
