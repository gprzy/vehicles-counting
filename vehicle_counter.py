import sys
import json
import time
import numpy as np
import cv2

class VehicleCounter():
    def __init__(self,
                video_path: str,
                params_path: str):

        with open(params_path, 'r') as f:
            params = f.read()
            params = json.loads(params)

        self.VIDEO_PATH = video_path
        self.VIDEO_FPS = params['video_fps']
        self.COUNT_LINE_POS_INIT = params['count_line_pos_init']
        self.COUNT_LINE_POS_END = params['count_line_pos_end']
        self.COUNT_LINE_COLOR = params['count_line_color']
        self.COUNT_LINE_THICKNESS = params['count_line_thickness']
        self.ON_DETECT_COUNT_LINE_COLOR = params['on_detec_count_line_color']
        self.TEXT_DISPLAY = params['text_display']
        self.TEXT_POS = params['text_pos']
        self.TEXT_COLOR = params['text_color']
        self.TEXT_THICKNESS = params['text_thickness']
        self.TEXT_FONTSCALE = params['text_fontscale']
        self.RECT_MIN_WIDTH = params['rect_min_width']
        self.RECT_MIN_HEIGHT = params['rect_min_height']
        self.PIXEL_OFFSET = params['pixel_offset']
        self.SAVE_OUTPUT_PATH = params['save_output_path']

        self.SLEEP_TIME = 1/(self.VIDEO_FPS*4)

        self.cap = cv2.VideoCapture(self.VIDEO_PATH)

        self.DETEC = []
        self.vehicles_count = 0
        self.subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

        if self.SAVE_OUTPUT_PATH:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            self.out = cv2.VideoWriter(
                self.SAVE_OUTPUT_PATH,
                fourcc,
                fps=30.0,
                frameSize=(width,  height)
            )

    # obtendo o centro da imagem
    def get_image_center(self, x, y, width, height):
        x1 = width // 2
        y1 = height // 2
        cx = x + x1
        cy = y + y1
        return cx, cy

    # aplicando vários filtros na imagem
    def apply_filters(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)

        sub = self.subtractor.apply(blur)
        
        dilated = cv2.dilate(sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
        morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

        contours, img = cv2.findContours(morph,
                                         cv2.RETR_TREE,
                                         cv2.CHAIN_APPROX_SIMPLE)

        return gray, blur, sub, dilated, kernel, morph, contours, img

    # incrementando o contador de veículos
    def increment_counter(self, detec, frame):
        for (x, y) in detec:
            if (self.COUNT_LINE_POS_INIT[1] + self.PIXEL_OFFSET) > y > (self.COUNT_LINE_POS_INIT[1] - self.PIXEL_OFFSET):
                self.vehicles_count += 1

                # desenhando a linha na detecção
                cv2.line(
                    img=frame,
                    pt1=self.COUNT_LINE_POS_INIT,
                    pt2=self.COUNT_LINE_POS_END,
                    color=self.ON_DETECT_COUNT_LINE_COLOR,
                    thickness=self.COUNT_LINE_THICKNESS
                )
                
                detec.remove((x, y))

    # método principal
    def count(self):
        while self.cap.isOpened():
            time.sleep(self.SLEEP_TIME)
            ret, frame = self.cap.read()

            # if frame is read correctly ret is True
            if not ret:
                print('Erro na leitura do arquivo!')
                break
            
            # vídeo original
            cv2.imshow('input video', frame)

            # aplicando alguns filtros
            gray, blur, sub, dilated, kernel, morph, contours, img = self.apply_filters(frame)

            # linha de detecção
            cv2.line(
                img=frame,
                pt1=self.COUNT_LINE_POS_INIT,
                pt2=self.COUNT_LINE_POS_END,
                color=self.COUNT_LINE_COLOR,
                thickness=self.COUNT_LINE_THICKNESS
            )

            for (i, c) in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(c)
                validar_contorno = (w >= self.RECT_MIN_WIDTH) and (h >= self.RECT_MIN_HEIGHT)
                if not validar_contorno:
                    continue

                # retângulo envolta dos veículos
                cv2.rectangle(
                    img=frame, 
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=(0, 255, 0),
                    thickness=2
                )
                
                center = self.get_image_center(x, y, w, h)
                self.DETEC.append(center)

            # contagem de veículos
            self.increment_counter(self.DETEC, frame)
            
            # escrevendo na tela o número de veículos 
            cv2.putText(
                img=frame,
                text=f'{self.TEXT_DISPLAY} {self.vehicles_count}',
                org=self.TEXT_POS,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=self.TEXT_FONTSCALE,
                color=self.TEXT_COLOR,
                thickness=self.TEXT_THICKNESS
            )

            # vídeo de detecção e subtração
            cv2.imshow('detection video', frame)
            cv2.imshow('subtraction video', morph)

            if self.SAVE_OUTPUT_PATH:
                self.out.write(frame)

            # saída
            if cv2.waitKey(1) == ord('q'):
                break

        if self.SAVE_OUTPUT_PATH:
            self.out.release()

        self.cap.release()
        cv2.destroyAllWindows()

        print('número de veículos detectados =', self.vehicles_count)

if __name__ == '__main__':
    video_path = sys.argv[1]

    try:
        params_path = sys.argv[2]
    except:
        params_path = './params/params.json'
    
    model = VehicleCounter(video_path=video_path, params_path=params_path)
    model.count()
