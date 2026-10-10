"""Generate GitHub-friendly SVG figures from actual detector outputs."""
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np
from detect import detect


def image_svg(image, x, y, width=500):
    # Encode horizontal runs as SVG rectangles; no external image dependencies.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, original_width = image.shape[:2]
    scale = width / original_width
    pieces = [f'<g transform="translate({x},{y}) scale({scale})">']
    paths = {}
    for row in range(height):
        pixels = image[row]
        starts = np.r_[0, np.flatnonzero(np.any(pixels[1:] != pixels[:-1], axis=1)) + 1]
        ends = np.r_[starts[1:], original_width]
        for start, end in zip(starts, ends):
            b, g, r = map(int, pixels[start])
            color = f'#{r:02x}{g:02x}{b:02x}'
            paths.setdefault(color, []).append(f'M{start},{row}h{end-start}v1h-{end-start}z')
    for color, commands in paths.items():
        pieces.append(f'<path fill="{color}" d="{"".join(commands)}"/>')
    pieces.append('</g>')
    return ''.join(pieces)


def save_panels(path, title, subtitle, panels):
    width, height = 1080, 900
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
             f'<title>{escape(title)}</title>', '<rect width="1080" height="900" fill="#101827"/>',
             '<g font-family="sans-serif" fill="#eef2ff">',
             f'<text x="30" y="45" font-size="27" font-weight="bold">{escape(title)}</text>',
             f'<text x="30" y="75" font-size="15" fill="#aabbd5">{escape(subtitle)}</text>']
    for i, (label, image) in enumerate(panels):
        x, y = 30 + (i % 2) * 520, 125 + (i // 2) * 390
        parts.append(f'<text x="{x}" y="{y-12}" font-size="18">{escape(label)}</text>')
        parts.append(image_svg(image, x, y))
    parts.append('</g></svg>')
    path.write_text(''.join(parts))


def scene(color=170, merged=False):
    image = np.full((350, 500, 3), 25, np.uint8)
    positions = (60, 130, 360) if merged else (60, 210, 360)
    for x in positions:
        cv2.rectangle(image, (x, 100), (x+65, 220), (color, color, color), -1)
    if merged:
        cv2.rectangle(image, (120, 120), (140, 200), (100, 100, 100), -1)
    return image


def main():
    folder = Path('docs/images')
    folder.mkdir(parents=True, exist_ok=True)
    image = scene()
    result, annotated, mask, edges = detect(image)
    save_panels(folder / 'pipeline.svg', 'Parking-Lot Vehicle Counting',
                'Synthetic demonstration | HSV segmentation, morphology, edges and contour counting',
                [('1. Input: three ideal vehicle regions', image), ('2. Foreground mask', mask),
                 ('3. Canny edges', edges), (f"4. Annotated output: {result['estimated_count']} estimated vehicles", annotated)])
    experiments = [('Separated regions', scene()), ('Dark regions', scene(35)),
                   ('Overexposed regions', scene(250)), ('Connected regions', scene(170, True))]
    panels, results = [], []
    for label, sample in experiments:
        outcome, output, _, _ = detect(sample)
        count = outcome['estimated_count']
        panels.append((f'{label}: estimated {count}, true 3', output))
        results.append(dict(scenario=label, true_count=3, estimated_count=count, count_error=count-3))
    save_panels(folder / 'sensitivity.svg', 'Threshold Sensitivity',
                'Synthetic checks with the same fixed configuration | These are not real-car accuracy results', panels)
    (folder.parent / 'synthetic_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
