import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Função para calcular a distância entre dois pontos
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Inicializar o detector de mãos da biblioteca cvzone
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Cores para os círculos
shape_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255)]

# Definir estados possíveis
STATES = ['aberto', 'fechado']

# Função para determinar o estado de um dedo
def determine_state(dist, threshold=100):
    if dist < threshold:
        return 'fechado'
    else:
        return 'aberto'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar mãos na imagem
    hands, frame = detector.findHands(frame)

    # Lista para armazenar os pontos das articulações da mão
    hand_points_list = []

    # Desenhar pontos-chave das articulações da mão na imagem e armazenar suas coordenadas
    if hands:
        for hand in hands:
            hand_points = hand["lmList"]  # Lista de pontos chave
            hand_type = hand["type"]  # Tipo de mão ("Left" ou "Right")

            for i, point in enumerate(hand_points):
                # Desenhar as pontas dos dedos com cores correspondentes
                if i in [4, 8, 12, 16, 20]:
                    color_index = [4, 8, 12, 16, 20].index(i)
                    cv2.circle(frame, (point[0], point[1]), 5, shape_colors[color_index], -1)
                else:
                    # Outros pontos
                    cv2.circle(frame, (point[0], point[1]), 5, (255, 0, 0), -1)

            # Calcular a posição média entre os nós 0, 1, 5, 9, 13 e 17
            x_sum = hand_points[0][0]
            y_sum = hand_points[0][1]
            for index in [1, 5, 9, 13, 17]:
                x_sum += hand_points[index][0]
                y_sum += hand_points[index][1]
            x_avg = x_sum // 6
            y_avg = y_sum // 6

            # Adicionar o ponto amarelo na posição média
            hand_points.append((x_avg, y_avg))
            cv2.circle(frame, (x_avg, y_avg), 30, (0, 255, 255), -1)  # Metade do tamanho do ponto amarelo

            # Verificar a distância entre a ponta de cada dedo e o ponto amarelo central
            for i, (finger_tip_index, color) in enumerate(zip([4, 8, 12, 16, 20], shape_colors)):
                finger_tip_point = hand_points[finger_tip_index]
                dist = distance(finger_tip_point, (x_avg, y_avg))

                # Determinar o estado do dedo
                state = determine_state(dist)

                if hand_type == "Left":
                    # Localização dos círculos no canto superior esquerdo
                    circle_x = 50
                    circle_y = 50 + i * 60
                else:
                    # Localização dos círculos no canto superior direito
                    circle_x = frame.shape[1] - 50
                    circle_y = 50 + i * 60

                # Desenhar círculos coloridos
                cv2.circle(frame, (circle_x, circle_y), 30, color, -1)

                # Escrever o estado de cada dedo
                if hand_type == "Left":
                    cv2.putText(frame, state, (circle_x + 40, circle_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, state, (circle_x - 90, circle_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            hand_points_list.append(hand_points)

    # Desenhar linhas seguindo a lógica da mão entre os pontos de articulação para ambas as mãos
    for hand_points in hand_points_list:
        if len(hand_points) > 0:
            for connection in [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]:
                start_point = (hand_points[connection[0]][0], hand_points[connection[0]][1])
                end_point = (hand_points[connection[1]][0], hand_points[connection[1]][1])
                cv2.line(frame, start_point, end_point, (0, 0, 255), 2)

    # Exibir o frame resultante
    cv2.imshow('Hand Pose Estimation', frame)

    # Sair do loop se a tecla 'esc' for pressionada
    key = cv2.waitKey(1)
    if key == 27:  # 27 é o código ASCII para a tecla 'esc'
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
