import numpy as np
import matplotlib.pyplot as plt

# Dimensões da imagem
colunas = 50
linhas = 25
image_matrix = np.zeros([linhas, colunas], dtype=np.uint8)

# Desenhar letra P
image_matrix[1:15, 2] = 120
image_matrix[1, 2:6] = 120
image_matrix[6:11, 2] = 120
image_matrix[1:6, 6] = 120

# Desenhar letra S
image_matrix[1, 9:13] = 120
image_matrix[2:6, 9] = 120
image_matrix[6, 9:13] = 120
image_matrix[7:11, 13] = 120
image_matrix[11, 9:13] = 120

# Desenhar letra N
for i in range(12):
    image_matrix[i, 16] = 120
    image_matrix[i, 16+i] = 120
    image_matrix[i, 28-i] = 120

# Desenhar letra M
for i in range(12):
    image_matrix[i, 32] = 120
    image_matrix[i, 29+i] = 120
    image_matrix[i, 35-i] = 120

# Exibir a imagem
plt.imshow(image_matrix, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.show()
