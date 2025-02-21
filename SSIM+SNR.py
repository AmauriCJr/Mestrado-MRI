import matplotlib.pyplot as plt
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import math
import os
import pandas as pd

def load_image(image_path):
    image = io.imread(image_path, as_gray=False) 

    if image.shape[-1] == 4: 
        image = image[:, :, :3]

    grayscale_image = np.mean(image, axis=-1)

    return grayscale_image

def SSIM(image1, image2):
    """Calcula o SSIM entre duas imagens convertidas corretamente para escala de cinza."""
    image1_gray = np.mean(image1, axis=-1) if image1.ndim == 3 else image1
    image2_gray = np.mean(image2, axis=-1) if image2.ndim == 3 else image2

    return compare_ssim(image1_gray, image2_gray, data_range=image2_gray.max() - image2_gray.min())

def SNR(image, radius_percentage=0.5):
    """Calcula a relação sinal-ruído (SNR) dentro de uma região circular central."""
    if image.ndim == 3: 
        image = np.mean(image, axis=-1)

    height, width = image.shape
    radius = math.floor(min(height, width) * radius_percentage)
    center_x, center_y = width // 2, height // 2

    mask = np.zeros_like(image)
    yy, xx = np.ogrid[:height, :width]
    mask_area = (xx - center_x)**2 + (yy - center_y)**2 <= radius**2
    mask[mask_area] = 1

    signal = np.std(image[mask == 1])
    noise = np.std(image[mask == 0])

    return 10 * np.log10((signal**2) / (noise**2))

def count_files(folder):
    """Conta quantos arquivos PNG existem na pasta."""
    return len([f for f in os.listdir(folder) if f.endswith('.png') and os.path.isfile(os.path.join(folder, f))])


file_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Generated Images\Image'
test_path = r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\datasets\birn_anatomical_part\test'

ssim_pr_array, ssim_in_array = [], []
snr_pr_array, snr_in_array = [], []

n = count_files((r'C:\Users\amaur\OneDrive\Área de Trabalho\Eletrônica\Mestrado\GANs\cgans\dev\mricganspace_150\Generated Images'))

for i in range(int(n/3)):
    img_n = i
    print(i)
    image1 = load_image(file_path + str(img_n) + '__Ground Truth.png')
    image2 = load_image(file_path + str(img_n) + '__Predicted Image.png')
    image3 = load_image(file_path + str(img_n) + '__Input Image.png')

    ssim_pr_array.append(SSIM(image1, image2))
    ssim_in_array.append(SSIM(image1, image3))
    snr_pr_array.append(SNR(image2))
    snr_in_array.append(SNR(image3))


print(f"SSIM input: {np.mean(ssim_in_array):.4f} ± {np.std(ssim_in_array):.4f}")
print(f"SNR input (dB): {np.mean(snr_in_array):.2f} ± {np.std(snr_in_array):.2f}")
print(f"SSIM predicted: {np.mean(ssim_pr_array):.4f} ± {np.std(ssim_pr_array):.4f}")
print(f"SNR predicted (dB): {np.mean(snr_pr_array):.2f} ± {np.std(snr_pr_array):.2f}")
