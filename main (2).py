import cv2
import numpy as np

def image_processing():
    # Task 1
    # Функция Отразить по горизонтали и перевернуть
    img = cv2.imread('images/Variant-7.jpg')
    cv2.imshow('Original', img)
    # отражаем изображение относительно горизонтали
    img_flip = cv2.flip(img, 0)
    # получим размеры изображения для поворота
    # и вычислим центр изображения
    (h, w) = img_flip.shape[:2]
    center = (w / 2, h / 2)
    # повернем изображение на 180 градусов
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(img_flip, M, (w, h))
    cv2.imshow('Task 1', rotated)

def video_processing(Mesure = False):
    # Task 2
    # Task 3 (Mesure = True)
    # matching template
    template = cv2.imread('ref-point.jpg',cv2.IMREAD_GRAYSCALE)
    (w_t, h_t) = template.shape[:2]
    cv2.imshow('template', template)
    # connect to camera
    cap = cv2.VideoCapture(0)
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert it to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if Mesure:
            (w, h) = frame.shape[:2]
            center = (h // 2, w // 2)
        # Perform match operations.
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        # Specify a threshold
        threshold = 0.85
        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= threshold)
        # Draw a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + w_t, pt[1] + h_t), (0, 255, 0), 1)
            if Mesure:
                center_t = (pt[0] + (w_t // 2), pt[1] + (h_t // 2))
                dis = ((center[0] - center_t[0]) ** 2 + (center[1] - center_t[1]) ** 2) ** 0.5
                cv2.putText(frame, 'Distance ' + str(int(dis)) + ' px', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                cv2.line(frame, center, center_t, (255,0,0), 1)
            break

        #out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #out.release()


if __name__ == '__main__':
    # Task 1
    #image_processing()
    # Task 2
    #video_processing()
    # Task 3
    video_processing(True)

cv2.waitKey(0)
cv2.destroyAllWindows()
