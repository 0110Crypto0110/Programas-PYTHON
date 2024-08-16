import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)

URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"
asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

classFile = "coco_class_labels.txt"

with open(classFile) as fp:
    labels = fp.read().split("\n")

modelFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

# Read the Tensorflow network
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

def detect_objects(net, im, dim=300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    objects = net.forward()
    return objects


import cv2
import numpy as np

# Função para detecção de objetos em uma imagem
def detect_objects_in_image(net, image, threshold=0.5):
    # Realiza a detecção de objetos na imagem
    objects = detect_objects(net, image)

    # Altura e largura da imagem
    image_height, image_width, _ = image.shape

    # Loop através dos objetos detectados
    for i in range(objects.shape[2]):
        # Obtém o ID da classe e a confiança da detecção
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Verifica se a confiança é maior que o limite definido
        if score > threshold:
            # Calcula as coordenadas do retângulo delimitador
            x_left_bottom = int(objects[0, 0, i, 3] * image_width)
            y_left_bottom = int(objects[0, 0, i, 4] * image_height)
            x_right_top = int(objects[0, 0, i, 5] * image_width)
            y_right_top = int(objects[0, 0, i, 6] * image_height)

            # Desenha o retângulo delimitador na imagem
            cv2.rectangle(image, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0), thickness=2)

            # Adiciona o nome do objeto e a confiança
            label = "{}: {:.2f}".format(labels[classId], score)
            cv2.putText(image, label, (x_left_bottom, y_left_bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

# Carrega a imagem
image_path = "images/criativo-65bd9444deb2aMDIvMDIvMjAyNCAyMmgxNw==.jpg"
image = cv2.imread(image_path)

# Realiza a detecção de objetos na imagem
detected_image = detect_objects_in_image(net, image)

# Exibe a imagem com os objetos detectados
cv2.imshow("Detected Objects", detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#---------------
'''
import cv2
import numpy as np

def detect_objects_in_image(net, image, threshold=0.5):
    # Realiza a detecção de objetos na imagem
    objects = detect_objects(net, image)

    # Altura e largura da imagem
    image_height, image_width, _ = image.shape

    # Loop através dos objetos detectados
    for i in range(objects.shape[2]):
        # Obtém o ID da classe e a confiança da detecção
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Verifica se a confiança é maior que o limite definido
        if score > threshold:
            # Calcula as coordenadas do retângulo delimitador
            x_left_bottom = int(objects[0, 0, i, 3] * image_width)
            y_left_bottom = int(objects[0, 0, i, 4] * image_height)
            x_right_top = int(objects[0, 0, i, 5] * image_width)
            y_right_top = int(objects[0, 0, i, 6] * image_height)

            # Desenha o retângulo delimitador na imagem
            cv2.rectangle(image, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0), thickness=2)

            # Adiciona o nome do objeto e a confiança
            label = "{}: {:.2f}".format(labels[classId], score)
            cv2.putText(image, label, (x_left_bottom, y_left_bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Define a área delimitada pelo retângulo
            roi = image[y_left_bottom:y_right_top, x_left_bottom:x_right_top]
            # Converte a região de interesse para escala de cinza
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Aplica a limiarização na região de interesse
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Encontra os contornos na imagem binarizada
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Desenha os pontos
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Desenha um círculo no ponto
                    cv2.circle(image, (x_left_bottom + cX, y_left_bottom + cY), 5, (255, 0, 0), -1)

    return image

# Carrega a imagem
image_path = "images/criativo-65bd9444deb2aMDIvMDIvMjAyNCAyMmgxNw==.jpg"
image = cv2.imread(image_path)

# Realiza a detecção de objetos na imagem
detected_image = detect_objects_in_image(net, image)

# Exibe a imagem com os objetos detectados e os pontos dentro da área delimitada pelo retângulo
cv2.imshow("Detected Objects", detected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#-----------------



#------------------------


'''

# Camera function from the first code
def camera_detection():
    source = cv2.VideoCapture(0)  # Use camera as source
    win_name = "Camera Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) != 27:
        has_frame, frame = source.read()
        if not has_frame:
            break
        frame = cv2.flip(frame, 1)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        objects = detect_objects(net, frame)  # Detect objects in the current frame

        for i in range(objects.shape[2]):
            classId = int(objects[0, 0, i, 1])
            score = float(objects[0, 0, i, 2])
            if score > 0.5:  # Adjust threshold as needed
                x_left_bottom = int(objects[0, 0, i, 3] * frame_width)
                y_left_bottom = int(objects[0, 0, i, 4] * frame_height)
                x_right_top = int(objects[0, 0, i, 5] * frame_width)
                y_right_top = int(objects[0, 0, i, 6] * frame_height)

                # Desenhar retângulo
                cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))

                # Adicionar nome do objeto
                label = "{}: {:.2f}".format(labels[classId], score)
                cv2.putText(frame, label, (x_left_bottom, y_left_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyAllWindows()

camera_detection()

'''

import cv2
import numpy as np

# Função para processar cada imagem
def process_image(image):
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Inicializar o detector de pontos de interesse
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    
    # Criar a nuvem de pontos usando os pontos de interesse
    point_cloud = np.zeros((len(p0), 3))
    for i, p in enumerate(p0):
        point_cloud[i] = [p[0][0], p[0][1], 0]  # A profundidade é definida como 0 aqui
    
    return point_cloud

# Carregar as imagens
image_paths = ["images/esfera-vermelha-d-119076254.jpg.webp]  # Substitua com os caminhos das suas imagens
images = [cv2.imread(path) for path in image_paths]

# Inicializar a nuvem de pontos
point_cloud = None

# Processar cada imagem
for image in images:
    # Processar a imagem
    image_point_cloud = process_image(image)
    
    # Adicionar os pontos da imagem à nuvem de pontos
    if point_cloud is None:
        point_cloud = image_point_cloud
    else:
        point_cloud = np.concatenate((point_cloud, image_point_cloud))

# Salvar ou visualizar a nuvem de pontos
# Você precisará de uma biblioteca específica para visualizar a nuvem de pontos, como Open3D ou Matplotlib 3D.
# Aqui está um exemplo com Matplotlib 3D:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
