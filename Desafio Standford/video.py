import cv2
import numpy as np
cap = cv2.VideoCapture('output.avi')  # Substitua 'seu_video.mp4' pelo caminho do seu vídeo

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Aplicar a subtração de fundo no quadro atual
    fgmask = fgbg.apply(frame)
    
    # Aplicar uma operação de limiarização para segmentar os objetos em movimento
    threshold = 50  # Ajuste esse valor conforme necessário
    _, thresh = cv2.threshold(fgmask, threshold, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos dos objetos em movimento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenhar retângulos ao redor dos objetos em movimento
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Ajuste essa área mínima conforme necessário
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Mostrar o quadro resultante
    cv2.imshow('Detecção de Movimento', frame)
    
    if cv2.waitKey(30) & 0xFF == 27:  # Pressione a tecla 'Esc' para sair
        break

cap.release()
cv2.destroyAllWindows()
