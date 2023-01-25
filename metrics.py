import numpy as np
from skimage.metrics import structural_similarity as _ssim
from skimage import img_as_float

def ssim(image_1, image_2):
    ssim = _ssim(img_as_float(image_1), img_as_float(image_2))
    return ssim

def ssim_mean(set_1, set_2):
    ssim = 0
    for image_1, image_2 in zip(set_1, set_2):
        ssim += _ssim(img_as_float(image_1), img_as_float(image_2))
    ssim = ssim/len(set_1)
    return ssim

def norm_1(image_1, image_2):
    distance = np.absolute(image_1 - image_2)
    distance = np.sum(distance)
    return distance

def norm_2(image_1, image_2):
    distance = (image_1 - image_2) ** 2
    distance = np.sum(distance)
    distance = np.sqrt(distance)
    return distance

def norm_1_mean(set_1, set_2):
    distance = np.absolute(set_1 - set_2)
    distance = np.sum(distance, axis=(1, 2))
    distance = np.mean(distance)
    return distance

def norm_2_mean(set_1, set_2):
    distance = (set_1 - set_2) ** 2
    distance = np.sum(distance, axis=(1, 2))
    distance = np.sqrt(distance)
    distance = np.mean(distance)
    return distance