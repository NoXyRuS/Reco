from flask import Flask, request, jsonify, Response, session
from flask_cors import CORS
import json
import threading
import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os
import imutils
import math
import time
import psycopg2
import base64

app = Flask(__name__)
app.secret_key = 't1+q7/kQ7bzrT3X2YAU7qBdaTb9Au3+yRkwAHZP2TxA='
CORS(app)



@app.route('/video_feed')
def video_feed():
    return Response(generar_video_biometrico(), mimetype='multipart/x-mixed-replace; boundary=frame')



# Configuración de la conexión a la base de datos
def conectar_bd():
    try:
        connection = psycopg2.connect(
            dbname='FiskView',
            user='postgres',
            password='admin123',
            host='localhost',
            port='5432'
        )
        return connection
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

# Función para obtener el usuario basado en la imagen
def obtener_usuario(imagen_facial):
    connection = conectar_bd()
    if connection is None:
        return None

    cursor = connection.cursor()

    if not isinstance(imagen_facial, np.ndarray):
        print("Error: imagen_facial no es un arreglo NumPy.")
        return None

    imagen_facial_rgb = cv2.cvtColor(imagen_facial, cv2.COLOR_BGR2RGB)

    _, imagen_buffer = cv2.imencode('.png', imagen_facial)
    imagen_base64 = base64.b64encode(imagen_buffer).decode('utf-8')

    cursor.execute("SELECT id_usuario, nombre, apellidos, dni, fecha_nacimiento, email, password, estado, imagen_facial FROM usuario_votante")
    usuarios = cursor.fetchall()

    for usuario in usuarios:
        id_usuario, nombre, apellidos, dni, fecha_nacimiento, email, password, estado, imagen_facial_bd = usuario

        imagen_facial_decodificada = base64.b64decode(imagen_facial_bd)
        np_array = np.frombuffer(imagen_facial_decodificada, np.uint8)
        imagen_bd = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        try:
            encoding_facial_rgb = fr.face_encodings(imagen_facial_rgb)[0]
            encoding_bd = fr.face_encodings(imagen_bd)[0]

            coincidencia = fr.compare_faces([encoding_bd], encoding_facial_rgb)

            if coincidencia[0]:
                cursor.close()
                connection.close()
                return id_usuario, nombre, apellidos  # Devuelve más información

        except IndexError:
            print("Error: No se encontró ningún rostro en una de las imágenes.")

    cursor.close()
    connection.close()
    return None

@app.route('/usuario_recibido', methods=['GET'])
def usuario_recibido():
    #global cut
    cut = np.load('temp.npy')

    if cut is None:
        return jsonify({"error": "No se ha encontrado rostro recortado."}), 400

    usuario = obtener_usuario(cut)

    if usuario:
        return jsonify({
            "status": "success",
            "id_usuario": usuario[0],
            "nombre": usuario[1],
            "apellidos": usuario[2]
        }), 200  # Cambia a 200 cuando se encuentra el usuario
    else:
        return jsonify({"error": "No se encontró el usuario."}), 404






# Paths
OuthFolderPathFace = r"C:\Users\bea_g\Desktop\Reconocimiento\Capturas"
PathUserCheck = ''  # Para verificar los rostros

# Variables
parpadeo = False
conteo = 0
muestra = 0
step = 0
capturas_guardadas = 0
limite_parpadeos = 10  # límite de parpadeos en una sesión

# Variables para obtener el rostro
offsety = 40
offsetx = 20

# Confianza de Detección
confThreshold = 0.5

# Malla Facial
mpDraw = mp.solutions.drawing_utils
ConfigDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Objeto de la Malla Facial
FacemeshObject = mp.solutions.face_mesh
FaceMesh = FacemeshObject.FaceMesh(max_num_faces=1)

# Detectar el objeto del Rostro
FaceObject = mp.solutions.face_detection
detector = FaceObject.FaceDetection(min_detection_confidence=0.5, model_selection=1)

def generar_video_biometrico():

    global conteo, parpadeo, step, capturas_guardadas

    # Capturar el Video
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        frameSave = frame.copy()
        frame = imutils.resize(frame, width=1280)

        # Convertir a RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret:
            # Inferencia de la Malla Facial
            res = FaceMesh.process(frameRGB)

            # Lista de Resultados
            px, py, lista = [], [], []
            if res.multi_face_landmarks:
                for rostros in res.multi_face_landmarks:
                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACEMESH_CONTOURS, ConfigDraw, ConfigDraw)

                    # Extraer Puntos Claves
                    for id, puntos in enumerate(rostros.landmark):
                        al, an, _ = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                    if len(lista) == 468:
                        x1, y1 = lista[374][1:]  # Ojo izquierdo
                        x2, y2 = lista[386][1:]  # Ojo derecho
                        longitud1 = math.hypot(x2 - x1, y2 - y1)

                        x3, y3 = lista[145][1:]  # Boca superior
                        x4, y4 = lista[159][1:]  # Boca inferior
                        longitud2 = math.hypot(x4 - x3, y4 - y3)

                        # Detección de Rostro
                        faces = detector.process(frameRGB)
                        if faces.detections is not None:
                            for face in faces.detections:
                                score = face.score[0]
                                bbox = face.location_data.relative_bounding_box
                                if score > confThreshold:
                                    xi = int(bbox.xmin * an - offsetx * bbox.width / 200)
                                    yi = int(bbox.ymin * al - offsety * bbox.height / 200)
                                    anc = int(bbox.width * an + offsetx * bbox.width / 100)
                                    alt = int(bbox.height * al + offsety * bbox.height / 100)
                                    xf, yf = xi + anc, yi + alt

                                    xi, yi, anc, alt = max(xi, 0), max(yi, 0), max(anc, 0), max(alt, 0)

                                    if step == 0:
                                        cv2.rectangle(frame, (xi, yi), (xf, yf), (255, 255, 255), 2)
                                        x5 = lista[33][1]  # Punto referencia izquierdo
                                        x6 = lista[133][1]  # Punto referencia derecho
                                        x7 = lista[362][1]  # Punto referencia izquierdo
                                        x8 = lista[263][1]  # Punto referencia derecho

                                        if x7 > x5 and x8 < x6:
                                            print("Mirando al frente")
                                        else:
                                            cv2.putText(frame, "Mirada desviada", (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                                                        0.8, (0, 0, 255), 2)

                                        if longitud1 <= 10 and longitud2 <= 10 and not parpadeo:
                                            conteo += 1
                                            parpadeo = True
                                        elif longitud1 > 10 and longitud2 > 10 and parpadeo:
                                            parpadeo = False

                                        cv2.putText(frame, f'Parpadeos: {conteo}', (1070, 375),
                                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                                        if conteo >= 3 and longitud1 > 15 and longitud2 > 15:
                                            timestamp = time.strftime("%Y%m%d-%H%M%S")

                                            # Ajustar el tamaño del recorte
                                            cut_width = int(anc * 1.2)  # Aumentar el ancho en un 20%
                                            cut_height = int(alt * 1.2)  # Aumentar la altura en un 20%

                                            # Ajustar las coordenadas del recorte
                                            cut_x = max(xi - int(cut_width * 0.1), 0)  # Desplazar a la izquierda
                                            cut_y = max(yi - int(cut_height * 0.1), 0)  # Desplazar hacia arriba

                                            cut = frameSave[cut_y:cut_y + cut_height, cut_x:cut_x + cut_width]

                                            np.save('temp', cut)
                                            #cut_str = np.array2string(cut)

                                            #with open('datos.txt', 'w') as f:
                                                #f.write(cut_str)


                                            # Obtener usuario basado en la imagen recortada
                                            usuario = obtener_usuario(cut)

                                            # Imprimir información del usuario
                                            if usuario:
                                                print(
                                                    f"Usuario encontrado: ID: {usuario[0]}, Nombre: {usuario[1]} {usuario[2]}")
                                            else:
                                                print("No se encontró el usuario.")

                                            # Guardar la imagen (opcional)
                                            if cv2.imwrite(
                                                    f'{OuthFolderPathFace}/face_capture_{timestamp}_{capturas_guardadas}.png',
                                                    cut):
                                                print("Imagen guardada exitosamente.")
                                            else:
                                                print("Error al guardar la imagen.")

                                            capturas_guardadas += 1
                                            step = 1
                                    else:
                                        conteo = 0

            # Codificar el frame en JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
    # Retornar id_usuario si fue encontrado



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
