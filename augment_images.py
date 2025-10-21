import os
import cv2
import re
import glob
import random 
import numpy as np
from natsort import natsorted
from skimage import feature, color, transform, util




def augment_image(image, filename):
    aug_images = []

    # 1. Horizontal Flip
    flipped = cv2.flip(image, 1)
    aug_images.append(("flip", flipped))

    # 2. Rotate by random angle (-25 to 25 degrees)
    angle = random.randint(-25, 25)
    rotated = transform.rotate(image, angle, mode='edge')
    rotated = (rotated * 255).astype(np.uint8)
    aug_images.append((f"rotate_{angle}", rotated))

    # 3. Add random noise
    noisy = util.random_noise(image)
    noisy = (noisy * 255).astype(np.uint8)
    aug_images.append(("noise", noisy))

    # 4. Brightness adjustment
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.2, 0, 255)
    brighter = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    aug_images.append(("bright", brighter))
    
    #5. main image
    aug_images.append(("main", image))
    
    for aug_type, aug_img in aug_images:
        save_path = os.path.join(aug_dir, f"{os.path.splitext(filename)[0]}_{aug_type}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

    return [aug_img for _, aug_img in aug_images]




#augment data and create an aug_images folder
image_paths = glob.glob('processed_images/*.jpg')
aug_dir = 'aug_images'
os.makedirs(aug_dir, exist_ok=True)
for path in image_paths:
    #load every image and augment it 
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filename = os.path.basename(path)
    aug_images = augment_image(img, filename)


