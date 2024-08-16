import cv2
import numpy as np
import open3d as o3d
import tkinter as tk
from PIL import Image, ImageTk

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Inicialização da câmera
        self.cap = cv2.VideoCapture(0)

        # Frame para exibir a imagem da câmera
        self.frame = tk.Label(window)
        self.frame.pack()

        # Botão para capturar imagens
        self.btn_capture = tk.Button(window, text="Capturar Imagem", width=20, command=self.capture_image)
        self.btn_capture.pack()

        # Botão para sair do programa
        self.btn_exit = tk.Button(window, text="Sair", width=20, command=self.close_program)
        self.btn_exit.pack()

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite("captured_image.png", frame)

    def close_program(self):
        self.cap.release()
        self.window.quit()

def detect_objects(net, im, dim=300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    objects = net.forward()
    return objects

def create_3d_model(images, objects_list, frame_width, frame_height):
    points = []

    for i, objects in enumerate(objects_list):
        for j in range(objects.shape[2]):
            classId = int(objects[0, 0, j, 1])
            score = float(objects[0, 0, j, 2])
            if score > 0.5:  # Adjust threshold as needed
                x_left_bottom = int(objects[0, 0, j, 3] * frame_width)
                y_left_bottom = int(objects[0, 0, j, 4] * frame_height)
                x_right_top = int(objects[0, 0, j, 5] * frame_width)
                y_right_top = int(objects[0, 0, j, 6] * frame_height)

                # Calculate 3D position of object center
                center_x = (x_left_bottom + x_right_top) / 2
                center_y = (y_left_bottom + y_right_top) / 2
                depth = 1.0  # Assume a fixed depth for simplicity
                center_z = depth * i  # Increment depth for each image

                # Add point to list of points
                points.append([center_x, center_y, center_z])

    # Convert list of points to numpy array
    points_array = np.array(points)

    # Create Open3D point cloud from numpy array
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_array)

    return point_cloud

def main():
    # Load SSD model and other configurations
    modelFile = "frozen_inference_graph.pb"
    configFile = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    # Create GUI window
    window = tk.Tk()
    app = App(window, "Capture and 3D Model")

    # Main loop to capture images
    images = []
    objects_list = []
    while True:
        ret, frame = app.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            app.frame.img = img
            app.frame.configure(image=img)

            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord("c"):
                images.append(frame.copy())
                objects = detect_objects(net, frame)
                objects_list.append(objects)

    # Dimensions of the frame
    frame_height, frame_width = images[0].shape[:2]

    # Create 3D model using captured images and detected objects
    point_cloud = create_3d_model(images, objects_list, frame_width, frame_height)

    # Visualize the 3D model
    o3d.visualization.draw_geometries([point_cloud])

    # Close the GUI window
    window.destroy()

if __name__ == "__main__":
    main()
