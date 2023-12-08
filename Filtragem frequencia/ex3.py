import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('sinc_original.png', cv2.IMREAD_GRAYSCALE)

# Calcula a Transformada de Fourier 2D da imagem
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)

# Tamanho da imagem
rows, cols = image.shape

# Espectro de magnitude
magnitude_spectrum = 20 * np.log(np.abs(f_transform_shifted))

# Definir a frequência de corte para os filtros passa-alta
D = 30  # Pode ajustar esse valor

# Filtro passa-alta do tipo ideal
ideal_high_pass_filter = np.ones((rows, cols), np.uint8)
center_row, center_col = rows // 2, cols // 2
ideal_high_pass_filter[center_row - D:center_row + D, center_col - D:center_col + D] = 0

# Filtro passa-alta de Butterworth
butterworth_high_pass_filter = 1 / (1 + (np.sqrt(2) - 1) * (np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ideal_high_pass_filter)))) ** 2)

# Filtro passa-alta Gaussiano
gaussian_high_pass_filter = 1 - np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ideal_high_pass_filter)))

# Aplicar os filtros
filtered_image_ideal = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted * ideal_high_pass_filter))
filtered_image_butterworth = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted * butterworth_high_pass_filter))
filtered_image_gaussian = np.fft.ifft2(np.fft.ifftshift(f_transform_shifted * gaussian_high_pass_filter))

# Visualizar os resultados
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(np.abs(filtered_image_ideal), cmap='gray')
plt.title('Filtro Passa-Alta Ideal'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(np.abs(filtered_image_butterworth), cmap='gray')
plt.title('Filtro Passa-Alta Butterworth'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(np.abs(filtered_image_gaussian), cmap='gray')
plt.title('Filtro Passa-Alta Gaussiano'), plt.xticks([]), plt.yticks([])

plt.show()
