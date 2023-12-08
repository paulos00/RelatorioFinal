import cv2
import numpy as np

# Carregando as imagens das placas
placa_sem_defeito = cv2.imread('pcbCroppedTranslated.png')
placa_com_defeito = cv2.imread('pcbCroppedTranslatedDefected.png')

# Verificando se as imagens foram carregadas corretamente
if placa_sem_defeito is None or placa_com_defeito is None:
    print("Erro ao carregar as imagens.")
else:
    # Convertendo as imagens para tons de cinza (para simplificar a subtração)
    placa_sem_defeito_gray = cv2.cvtColor(placa_sem_defeito, cv2.COLOR_BGR2GRAY)
    placa_com_defeito_gray = cv2.cvtColor(placa_com_defeito, cv2.COLOR_BGR2GRAY)

    # Calculando a diferença entre as duas imagens
    diferenca = cv2.absdiff(placa_sem_defeito_gray, placa_com_defeito_gray)

    # Definindo um limite para destacar as diferenças (pode ser ajustado)
    limite = 30
    _, diferenca_binaria = cv2.threshold(diferenca, limite, 255, cv2.THRESH_BINARY)

    # Encontrando os contornos das diferenças
    contornos, _ = cv2.findContours(diferenca_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhando os contornos nas placas com defeito
    placa_com_defeito_copy = placa_com_defeito.copy()
    cv2.drawContours(placa_com_defeito_copy, contornos, -1, (0, 0, 255), 2)

    # Exibindo as imagens
    cv2.imshow('Placa sem defeito', placa_sem_defeito)
    cv2.imshow('Placa com defeito', placa_com_defeito_copy)
    cv2.waitKey(0)
    #fechando as imagens
    cv2.destroyAllWindows()
