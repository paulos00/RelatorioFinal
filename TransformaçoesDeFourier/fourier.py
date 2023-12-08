import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# Função para calcular a Transformada de Fourier
def calcular_transformada_fourier(sinal):
    N = len(sinal)
    frequencias = np.fft.fftfreq(N)
    espectro = np.fft.fft(sinal)
    return frequencias, espectro

# Função para calcular a Transformada Inversa de Fourier
def calcular_transformada_inversa_fourier(espectro):
    sinal_reconstruido = np.fft.ifft(espectro)
    return sinal_reconstruido

# Gerar um sinal de exemplo
t = np.linspace(0, 1, 1000, endpoint=False)  # Vetor de tempo de 0 a 1 segundos
frequencia_1 = 5  # Frequência da primeira imagem
frequencia_2 = 10  # Frequência da segunda imagem

# Criar 5 sinais de exemplo com diferentes frequências
sinais = []
for i in range(5):
    sinal = np.sin(2 * np.pi * frequencia_1 * t) + 0.5 * np.sin(2 * np.pi * frequencia_2 * t)
    sinais.append(sinal)

# Calcular a Transformada de Fourier e plotar o espectro e a fase para cada sinal
plt.figure(figsize=(12, 15))
for i, sinal in enumerate(sinais, 1):
    frequencias, espectro = calcular_transformada_fourier(sinal)
    
    plt.subplot(5, 2, 2*i-1)
    plt.plot(t, sinal)
    plt.title(f'Sinal {i}: Original')
    
    plt.subplot(5, 2, 2*i)
    plt.plot(frequencias, np.abs(espectro))
    plt.title(f'Sinal {i}: Espectro de Amplitude')
    plt.xlabel('Frequência (Hz)')

plt.tight_layout()
plt.show()
