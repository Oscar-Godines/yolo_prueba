import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def detectar_objetos(frame):
    results = model(frame)  
    return results


def detectar_movimiento(frame_actual, frame_anterior):
 
    gris_actual = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
    gris_anterior = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
    

    diferencia = cv2.absdiff(gris_actual, gris_anterior)
    

    _, umbral = cv2.threshold(diferencia, 25, 255, cv2.THRESH_BINARY)
    

    movimiento_detectado = np.sum(umbral) > 5000  
    
    return movimiento_detectado


cap = cv2.VideoCapture(0)

frame_anterior = None

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
 
    resultados = detectar_objetos(frame)
    

    resultados.render()
    

    if frame_anterior is not None:
        movimiento = detectar_movimiento(frame, frame_anterior)
        if movimiento:
            cv2.putText(frame, "Detecciones", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    

    cv2.imshow('Detección de Objetos', frame)
    

    frame_anterior = frame
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
