import os
import cv2
import numpy as np
from zipfile import ZipFile
from urllib.request import urlretrieve

# Função para baixar e descompactar ativos
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)

# URL para baixar os ativos
URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"
asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# Baixar os ativos se o arquivo zip não existir
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

# Inicialização da câmera
source = cv2.VideoCapture(0)
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Carregar o modelo de detecção de pose
protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = os.path.join("model", "pose_iter_160000.caffemodel")
net_pose = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
nPoints = 15
POSE_PAIRS = [
    [0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
    [1, 14], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]
]

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Detecção de pose
    blob_pose = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=True, crop=False)
    net_pose.setInput(blob_pose)
    output = net_pose.forward()

    # Desenhar pontos e esqueleto
    points = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = int((frame_width * point[0]) / output.shape[3])
        y = int((frame_height * point[1]) / output.shape[2])
        if prob > 0.1:
            cv2.circle(frame, (x, y), 8, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
        points.append((x, y))

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (255, 255, 0), 2)
            cv2.circle(frame, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

    cv2.imshow(win_name, frame)

source.release()
cv2.destroyAllWindows()
