from ultralytics import YOLO
import cv2
from info import generate_info
import threading
import time


last_plant_detected = None
cooldown = 3
last_time = 0

def llm_part(plant_name):
    info = generate_info(plant_name)
    print("\n   {{{PLANT INFO}}}   ")
    print(f"detected: {plant_name}")
    print(info)



def main():

    global last_plant_detected, last_time

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


        #detection part of the name
        if len(results[0].boxes) > 0:
            cls = results[0].boxes.cls[0].item()
            plant_name = results[0].names[int(cls)]
            #print(f"\n Detected plant: {plant_name}")

            #Triggering LLM when a new plant gets detected
            now = time.time()
            if plant_name != last_plant_detected and now - last_time > cooldown:
                last_plant_detected = plant_name
                last_time = now

                print(f"\n[+] New plant detected: {plant_name}")
                threading.Thread(target=llm_part, args=(plant_name,), daemon=True).start()

            #information = generate_info(plant_name)
            #print("\n Plant inforamtion:\n", information)
            #print("-" * 60)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

