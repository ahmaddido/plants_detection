from ultralytics import YOLO
import cv2

def main():

    model = YOLO("best.pt")

    cap = cv2.VideoCapture(0)
   
    while True:
        ret, frame = cap.read()

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.7,
            verbose=False
        )

        box_frame = results[0].plot()
        cv2.imshow("plants", box_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()