"""Evaluate total vehicle counts on an explicitly labelled image list."""
import argparse
import csv
import json
from pathlib import Path

import cv2
from detect import detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True, help='CSV: image,count; image paths relative to CSV')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation.json'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else None
    rows = []
    with args.labels.open(newline='') as file:
        for row in csv.DictReader(file):
            image_path = args.labels.parent / row['image']
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f'Cannot read {image_path}')
            truth = int(row['count'])
            if truth < 0:
                raise ValueError('Ground-truth counts must be nonnegative')
            result, _, _, _ = detect(image, config)
            predicted = result['estimated_count']
            rows.append(dict(image=row['image'], truth=truth, predicted=predicted, error=predicted-truth))
    if not rows:
        raise ValueError('Labels CSV is empty')
    summary = dict(images=len(rows), mean_absolute_count_error=sum(abs(r['error']) for r in rows)/len(rows),
                   exact_count_fraction=sum(r['error'] == 0 for r in rows)/len(rows), rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
