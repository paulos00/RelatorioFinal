import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Valores do r (taxa de crescimento)
img = Image.open('D:\Processamento digital de sinais\DigitalImageProcessingSamples\Images\enhace.png')
img = np.linspace(0, 10, 100)
# Diferentes valores para o parâmetro c
c_values = [0.5, 1, 2]

# Plot para cada valor de c
for c in c_values:
    s = c * np.log(1 + r)
    plt.plot(r, s, label=f'c = {c}')

plt.xlabel('r (Taxa de Crescimento)')
plt.ylabel('s')
plt.title('Transformação Logarítmica: s = c * log(1 + r)')
plt.legend()
plt.grid(True)
plt.show()
