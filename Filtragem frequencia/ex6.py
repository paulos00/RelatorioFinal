import cv2
import numpy as np
from scipy import ndimage

# Carregando a imagem
imagem = cv2.imread('sinc_original.png', cv2.IMREAD_GRAYSCALE)


f1 = 30  # Frequência de corte inferior
f2 = 70  # Frequência de corte superior


filtro_passa_banda = ndimage.gaussian_filter(imagem, sigma=f2/5) - ndimage.gaussian_filter(imagem, sigma=f1/5)


imagem_filtrada = (filtro_passa_banda - np.min(filtro_passa_banda)) / (np.max(filtro_passa_banda) - np.min(filtro_passa_banda)) * 255

imagem_filtrada = imagem_filtrada.astype(np.uint8)

# Exiba a imagem original e a imagem filtrada
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Imagem Filtrada', imagem_filtrada)

cv2.waitKey(0)
cv2.destroyAllWindows()