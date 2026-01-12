from ultralytics import YOLO
import cv2
from info import plant_q

def main():

    model = YOLO("best.pt")
    cap = cv2.VideoCapture(0)

    #solving the flickering detecion. it triggers onlu when stable
    last_name = None
    stable_count = 0
    triggered = set()

    while True:
        ret, frame = cap.read()
        
        #if not ret:
        #    continue

        results = model.predict(source=frame, imgsz=640, conf=0.7, verbose=False)
        cv2.imshow("plants", results[0].plot())

        if len(results[0].boxes) > 0:
            cls = int(results[0].boxes.cls[0].item())
            plant_name = results[0].names[cls]

            #to check if the same plant is across frames
            if plant_name == last_name:
                stable_count += 1
            else:
                last_name = plant_name
                stable_count = 1

            if stable_count >= 10 and plant_name not in triggered:
                triggered.add(plant_name)
                print(f"Plant detected: {plant_name}")
                plant_q.put(plant_name)

                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

