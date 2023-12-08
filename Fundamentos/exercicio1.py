import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\lena_gray_512.tif')
    print(img)
    img_array= np.array(img)
    negative_image = 255 - img_array

    negative_image = Image.fromarray(negative_image.astype('uint8'))

    plt.figure(figsize=(10,5))
 

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(negative_image, cmap='gray')
    plt.title('Imagem negativa')

    plt.show()
    img2 = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\house.tif')
    print(img2)
    img_array2= np.array(img2)
    negative_image2 = 255 - img_array2

    negative_image2 = Image.fromarray(negative_image2.astype('uint8'))

    plt.figure(figsize=(10,5))
 

    plt.subplot(1,2,1)
    plt.imshow(img2, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(negative_image2, cmap='gray')
    plt.title('Imagem negativa')
    
    img3 = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\cameraman.tif')
    print(img3)
    img_array3= np.array(img3)
    negative_image3 = 255 - img_array3

    negative_image3 = Image.fromarray(negative_image3.astype('uint8'))

    plt.figure(figsize=(10,5))
 

    plt.subplot(1,2,1)
    plt.imshow(img3, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(negative_image3, cmap='gray')
    plt.title('Imagem negativa')

    plt.show()








if __name__ == "__main__":
    main()
