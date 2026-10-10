"""Classical parked-vehicle counting baseline derived from Image Analysis Task 6."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULTS = dict(hsv_lower=[0, 0, 50], hsv_upper=[179, 60, 240],
                median_kernel=5, morphology_kernel=5, min_area=1700,
                normal_area=8500, merged_ratio=1.5, canny_lower=150, canny_upper=220)


def detect(image, config=None):
    cfg = {**DEFAULTS, **(config or {})}
    for key in ('median_kernel', 'morphology_kernel'):
        if not isinstance(cfg[key], int) or cfg[key] < 1 or cfg[key] % 2 == 0:
            raise ValueError(f'{key} must be a positive odd integer')
    if cfg['min_area'] <= 0 or cfg['normal_area'] <= 0 or cfg['merged_ratio'] <= 0:
        raise ValueError('Area thresholds and merged_ratio must be positive')
    lower, upper = np.asarray(cfg['hsv_lower']), np.asarray(cfg['hsv_upper'])
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower < 0) or np.any(lower > upper) or np.any(upper > [179, 255, 255]):
        raise ValueError('Invalid OpenCV HSV bounds')
    blurred = cv2.medianBlur(image, cfg['median_kernel'])
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower.astype(np.uint8), upper.astype(np.uint8))
    k = cfg['morphology_kernel']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(mask, cfg['canny_lower'], cfg['canny_upper'])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg['min_area']:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        count = 1 if w * h <= cfg['normal_area'] * cfg['merged_ratio'] else 2
        regions.append(dict(box=[x, y, w, h], contour_area=float(area), estimated_count=count))
    regions.sort(key=lambda region: (region['box'][1], region['box'][0]))
    output = image.copy()
    for region in regions:
        x, y, w, h = region['box']
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output, str(region['estimated_count']), (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
    total = sum(region['estimated_count'] for region in regions)
    cv2.putText(output, f'Estimated vehicles: {total}', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    return dict(estimated_count=total, regions=regions, config=cfg), output, mask, edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f'Cannot read image: {args.image}')
    config = json.loads(args.config.read_text()) if args.config else None
    result, annotated, mask, edges = detect(image, config)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('annotated', annotated), ('mask', mask), ('edges', edges)]:
        if not cv2.imwrite(str(args.output / f'{name}.png'), data):
            raise OSError(f'Could not save {name}')
    result['image'] = str(args.image)
    (args.output / 'detections.json').write_text(json.dumps(result, indent=2))
    print(f"Estimated vehicles: {result['estimated_count']}")


if __name__ == '__main__':
    main()
