import flet as ft
import cv2
import mediapipe as mp
from pickle import load
import numpy as np
import dictionary
import base64


# open the model
model_dict = load(open("./model.p", "rb"))
model = model_dict["model"]

# open the mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# open the webcam
cap = cv2.VideoCapture(0)

# open the labels dictionary
labels_dict = dictionary.labels_dict


class Countdown(ft.UserControl):
    def __init__(self):
        super().__init__()

    def did_mount(self):
        self.update_timer()

    def update_timer(self):
        while True:
            data_aux = []
            x_ = []
            y_ = []
            _, frame = cap.read()
            # autosize the frame
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)

                x1 = int(min(x_) * W) - 40
                y1 = int(min(y_) * H) - 40

                x2 = int(max(x_) * W) + 40
                y2 = int(max(y_) * H) + 40

                try:
                    prediction = model.predict([np.asarray(data_aux)])

                    predicted_character = labels_dict[int(prediction[0])]

                    presicion = model.predict_proba([np.asarray(data_aux)])

                    label = "{}: {}".format(
                        predicted_character, presicion[0][int(prediction[0])]
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )
                except:
                    pass

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

            # encode the frame
            _, buffer = cv2.imencode(".jpg", frame)
            img_base64 = base64.b64encode(buffer)
            self.img.src_base64 = img_base64.decode("utf-8")

            self.update()

    def build(self):
        self.img = ft.Image(border_radius=ft.border_radius.all(20))
        return self.img


section = ft.Container(
    margin=ft.margin.only(bottom=40),
    content=ft.Row(
        [
            ft.Card(
                elevation=30,
                content=ft.Container(
                    bgcolor=ft.colors.WHITE24,
                    padding=10,
                    border_radius=ft.border_radius.all(20),
                    content=ft.Column(
                        [
                            Countdown(),
                        ]
                    ),
                ),
            ),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
    ),
)


def main(page: ft.Page):
    page.title = "Sign Language Recognition"
    page.padding = 50

    page.add(
        section,
    )


if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=8000)
    cap.release()
    cv2.destroyAllWindows()
