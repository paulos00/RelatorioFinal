import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def apply_median_filter_numpy(image):
    height, width = image.shape
    
    # Calculando o tamanho da saída após a filtragem
    out_height = height - 2
    out_width = width - 2
    
    # Inicializando a matriz de saída
    output = np.zeros((out_height, out_width), dtype=np.uint8)
    
    # Aplicando o filtro de mediana
    for i in range(out_height):
        for j in range(out_width):
            patch = image[i:i+3, j:j+3]
            output[i, j] = np.median(patch)
    
    return output

def main_median_numpy():
    image = np.array(Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\lena_gray_512.tif').convert("L"))
    
    filtered_image = apply_median_filter_numpy(image)
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagem original')

    plt.subplot(1,2,2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Imagem filtrada')

    plt.show()
    

if __name__ == "__main__":
    main_median_numpy()
