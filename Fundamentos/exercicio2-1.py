import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
#import opencv as cv2
from scipy import ndimage

def apply_average_filter(image):
    kernel = np.ones((3, 3)) / 9
    filtered_image = ndimage.convolve(image, kernel, mode='reflect')
    return filtered_image.astype(np.uint8)

def main():
    img = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\House.tif').convert("L")  # Convert image to grayscale
    print(img.format)
    print(img.size)
    print(img.mode)
    
    npImg = np.array(img)
    
    # filtro com Pillow
    filtered_img_pillow = img.filter(ImageFilter.Kernel((3, 3), [1, 1, 1, 1, 1, 1, 1, 1, 1], scale=1/9))
    
    # filtro com  OpenCV
    kernel = np.ones((3, 3), dtype=np.float32) / 9
  #  filtered_img_opencv = cv2.filter2D(npImg, -1, kernel)
    
    # filtro com  NumPy
    filtered_img_numpy = apply_average_filter(npImg)
    
    # filtro com  SciPy
    filtered_img_scipy = apply_average_filter(npImg)
    
    # plotando com matplotlib
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    ax[0, 0].imshow(npImg, cmap='gray')
    ax[0, 0].set_title("Original Image")
    ax[0, 1].imshow(filtered_img_pillow, cmap='gray')    
    ax[0, 1].set_title("Imagem filtrada (Pillow)")
   # ax[0, 2].imshow(filtered_img_opencv, cmap='gray')    
    ax[0, 2].set_title("Imagem filtrada (OpenCV)")
    ax[1, 0].imshow(filtered_img_numpy, cmap='gray')    
    ax[1, 0].set_title("Imagem filtrada (NumPy)")
    ax[1, 1].imshow(filtered_img_scipy, cmap='gray')    
    ax[1, 1].set_title("Imagem filtrada (SciPy)")
    
    for row in ax:
        for col in row:
            col.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    img = Image.open('house.tif').convert("L")  # Convert image to grayscale
    print(img.format)
    print(img.size)
    print(img.mode)
    
    npImg = np.array(img)
    
    # filtro com Pillow
    filtered_img_pillow = img.filter(ImageFilter.Kernel((3, 3), [1, 1, 1, 1, 1, 1, 1, 1, 1], scale=1/9))
    
    # filtro com  OpenCV
    kernel = np.ones((3, 3), dtype=np.float32) / 9
    #filtered_img_opencv = cv2.filter2D(npImg, -1, kernel)
    
    # filtro com  NumPy
    filtered_img_numpy = apply_average_filter(npImg)
    
    # filtro com  SciPy
    filtered_img_scipy = apply_average_filter(npImg)
    
    # plotando com matplotlib
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    ax[0, 0].imshow(npImg, cmap='gray')
    ax[0, 0].set_title("Original Image")
    ax[0, 1].imshow(filtered_img_pillow, cmap='gray')    
    ax[0, 1].set_title("Imagem filtrada (Pillow)")
    #ax[0, 2].imshow(filtered_img_opencv, cmap='gray')    
    ax[0, 2].set_title("Imagem filtrada (OpenCV)")
    ax[1, 0].imshow(filtered_img_numpy, cmap='gray')    
    ax[1, 0].set_title("Imagem filtrada (NumPy)")
    ax[1, 1].imshow(filtered_img_scipy, cmap='gray')    
    ax[1, 1].set_title("Imagem filtrada (SciPy)")
    
    for row in ax:
        for col in row:
            col.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    img = Image.open('house.tif').convert("L")  # Convert image to grayscale
    print(img.format)
    print(img.size)
    print(img.mode)
    
    npImg = np.array(img)
    
    # filtro com Pillow
    filtered_img_pillow = img.filter(ImageFilter.Kernel((3, 3), [1, 1, 1, 1, 1, 1, 1, 1, 1], scale=1/9))
    
    # filtro com  OpenCV
    kernel = np.ones((3, 3), dtype=np.float32) / 9
    #filtered_img_opencv = cv2.filter2D(npImg, -1, kernel)
    
    # filtro com  NumPy
    filtered_img_numpy = apply_average_filter(npImg)
    
    # filtro com  SciPy
    filtered_img_scipy = apply_average_filter(npImg)
    
    # plotando com matplotlib
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    ax[0, 0].imshow(npImg, cmap='gray')
    ax[0, 0].set_title("Original Image")
    ax[0, 1].imshow(filtered_img_pillow, cmap='gray')    
    ax[0, 1].set_title("Imagem filtrada (Pillow)")
    #ax[0, 2].imshow(filtered_img_opencv, cmap='gray')    
    ax[0, 2].set_title("Imagem filtrada (OpenCV)")
    ax[1, 0].imshow(filtered_img_numpy, cmap='gray')    
    ax[1, 0].set_title("Imagem filtrada (NumPy)")
    ax[1, 1].imshow(filtered_img_scipy, cmap='gray')    
    ax[1, 1].set_title("Imagem filtrada (SciPy)")
    
    for row in ax:
        for col in row:
            col.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()