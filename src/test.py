from ultralytics import YOLO
import cv2
#import threading
#import time
from info import generate_info

last_detected = None
cooldown = 3  # seconds
last_time = 0

#def llm_worker(plant_name):
#    info = generate_info(plant_name)
    #print("\n========== PLANT INFO ==========")
    #print(f"Detected: {plant_name}")
    #print(info)
    #print("================================\n")


def main():

    model = YOLO("../data/best.pt")

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


        if len(results[0].boxes) > 0:
            cls = results[0].boxes.cls[0].item()                       # class index
            plant_name = results[0].names[int(cls)]                    # class label
            print(f"\nDetected plant: {plant_name}")

            # ➤ CALL LLM TO GET INFO
            info = generate_info(plant_name)
            print("\nPlant information:\n", info)
            #print("-" * 60)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()