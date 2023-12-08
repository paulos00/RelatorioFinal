from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def main():
    img = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\lena_gray_512.tif')
    npImg = np.array(img)

    npImg[0:10,0:10] = 255
    npImg[0:10,-10:] = 255
    npImg[-10:,0:10] = 255
    npImg[-10:,-10:] = 255

    imgWSquares = Image.fromarray(npImg)

    
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(imgWSquares, cmap='gray')
    plt.title('Imagem com quadrados brancos')

    plt.show()
    
    img2 = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\cameraman.tif')
    npImg2 = np.array(img2)

    npImg2[0:10,0:10] = 255
    npImg2[0:10,-10:] = 255
    npImg2[-10:,0:10] = 255
    npImg2[-10:,-10:] = 255

    imgWSquares2 = Image.fromarray(npImg2)

    
    plt.subplot(1,2,1)
    plt.imshow(img2, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(imgWSquares2, cmap='gray')
    plt.title('Imagem com quadrados brancos')

    plt.show()
    
    img3 = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\house.tif')
    npImg3 = np.array(img3)

    npImg3[0:10,0:10] = 255
    npImg3[0:10,-10:] = 255
    npImg3[-10:,0:10] = 255
    npImg3[-10:,-10:] = 255

    imgWSquares3 = Image.fromarray(npImg3)

    
    plt.subplot(1,2,1)
    plt.imshow(img3, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(imgWSquares3, cmap='gray')
    plt.title('Imagem com quadrados brancos')

    plt.show()
    
    
    
if __name__ == "__main__":
    main()