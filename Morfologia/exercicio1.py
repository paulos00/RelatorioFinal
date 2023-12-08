import cv2
import numpy as np
from matplotlib import pyplot as plt

# Função para realizar abertura
def abrir_imagem(imagem, elemento_estruturante):
    abertura = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, elemento_estruturante)
    return abertura

# Função para realizar fechamento
def fechar_imagem(imagem, elemento_estruturante):
    fechamento = cv2.morphologyEx(imagem, cv2.MORPH_CLOSE, elemento_estruturante)
    return fechamento

# Leitura da imagem
imagem = cv2.imread('Imagem1.tif', cv2.IMREAD_GRAYSCALE)

# Elemento estruturante
elemento_estruturante = np.array([[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]], np.uint8)

# Aplicando abertura
imagem_abertura = abrir_imagem(imagem, elemento_estruturante)

# Aplicando fechamento
imagem_fechamento = fechar_imagem(imagem, elemento_estruturante)

# Exibindo as imagens original, abertura e fechamento
plt.subplot(1, 3, 1), plt.imshow(imagem, 'gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(imagem_abertura, 'gray'), plt.title('Abertura')
plt.subplot(1, 3, 3), plt.imshow(imagem_fechamento, 'gray'), plt.title('Fechamento')

plt.show()