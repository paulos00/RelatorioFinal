from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def main():
    img = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\lena_gray_512.tif')
    img_array = np.array(img)

    half_pixels = img_array / 2

    half_pixels = Image.fromarray(half_pixels.astype('uint8'))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(half_pixels, cmap='gray')
    plt.title('Imagem com metade dos pixels')

    plt.show()

    img2 = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\cameraman.tif')
    img_array2 = np.array(img2)

    half_pixels2 = img_array2 / 2

    half_pixels2 = Image.fromarray(half_pixels2.astype('uint8'))

    plt.subplot(1,2,1)
    plt.imshow(img2, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(half_pixels2, cmap='gray')
    plt.title('Imagem com metade dos pixels')

    plt.show()

    img3 = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\house.tif')
    img_array3 = np.array(img3)

    half_pixels3 = img_array3 / 2

    half_pixels3 = Image.fromarray(half_pixels3.astype('uint8'))

    plt.subplot(1,2,1)
    plt.imshow(img3, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(half_pixels3, cmap='gray')
    plt.title('Imagem com metade dos pixels')

    plt.show()
if __name__ == "__main__":
    main()
