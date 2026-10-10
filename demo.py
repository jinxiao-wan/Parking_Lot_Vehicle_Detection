"""Create a synthetic sanity-check image, not a real-world benchmark."""
from pathlib import Path
import cv2
import numpy as np
from detect import detect

folder = Path('outputs/demo')
folder.mkdir(parents=True, exist_ok=True)
image = np.full((350, 500, 3), 25, dtype=np.uint8)
for x in (60, 210, 360):
    cv2.rectangle(image, (x, 100), (x+65, 220), (170, 170, 170), -1)
result, annotated, mask, edges = detect(image)
assert result['estimated_count'] == 3, result
assert detect(np.zeros_like(image))[0]['estimated_count'] == 0
cv2.imwrite(str(folder / 'input.png'), image)
cv2.imwrite(str(folder / 'annotated.png'), annotated)
cv2.imwrite(str(folder / 'mask.png'), mask)
(folder / 'labels.csv').write_text('image,count\ninput.png,3\n')
print('Synthetic checks passed: three separated regions and an empty image.')
