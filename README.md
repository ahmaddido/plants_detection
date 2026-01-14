# plants_detection_LLM

This project demonstrates how an intelligent plant detector can work by combining computer vision and large language models (LLMs). The system detects plants using a camera, identifies their names with bounding boxes, and provides useful information about each plant, such as:
 - Plant name (common, scientific, and German)
 - Growing instructions
 - Lighting and watering conditions
 - Toxicity


The entire system is designed to run locally, without requiring an internet connection.

## System Architecture

The project has four main layers:

1. Perception Layer
- Captures camera input
- Runs YOLOv11n for real-time plant detection
- Extracts the detected plant class name

2. Logic Layer
- Filters detections to avoid duplicate processing
- Triggers the LLM only when a new plant is detected
- Pushes the plant class name into a queue

3. Knowledge layer
- local JSON knowledge base gets loaded
- Runs the LLM in the background
- Generates plant explanations

4. Resutls
- Outputs


## Method

The first step in this project was data preparation. To build a custom plant detector, photos were taken of six different houseplants under various conditions. These images were then uploaded to Roboflow, where the plants were manually labeled. After completing the labeling process, the dataset was used to train a YOLOv11n model. The trained model was tested in real-time to verify its detection accuracy and performance. The implementation for real-time plant detection can be found in the (plant_detection.py) file.

<img width="767" height="453" alt="Image" src="https://github.com/user-attachments/assets/cac91fc7-35b1-47fb-b934-04786fc7030a" />


The next step was integrating a LLM into the system. Since the main goal of the project was to run the entire pipeline locally without any internet connection, Ollama was chosen as the LLM runtime environment. Ollama is an open-source tool that simplifies managing and running LLMs on local machines. It can be installed from:

https://ollama.com

Using Ollama allows the system to generate responses completely offline. 

To ensure controlled and reliable LLM outputs, a structured knowledge base was created in a JSON file. This file contains specific information for each plant, including scientific, common, and German names, lighting and watering conditions, growth instructions, and toxicity information. By relying on this predefined knowledge base, the LLM generates responses based on verified data. This ensure that the LLM stayes under control. This knowledge base is stored in the (information.json) file.

After preparing the knowledge base, the required LLM models were downloaded using Ollama. This was done by selecting a suitable model from the Ollama model library and pulling it via the terminal using:

```
ollama pull <model_name>
```

Two models were tested in this project: phi3:latest and qwen3:0.6b-q8_0. While phi3:latest provides more accurate, detailed, and well-structured responses, its larger size makes it computationally heavier on the CPU. In contrast, qwen3:0.6b-q8_0 is significantly smaller and offers faster and smoother performance, making it more suitable for real-time usage on limited hardware. Once the models were defined, a custom prompt was designed to precisely control the behavior of the LLM. The prompt definition and LLM configuration can be found in the (info.py) file.


## Faced issues
- Running YOLOv11 and the LLM simultaneously on the same machine caused the object detection process to freeze and significantly delayed LLM responses. This issue was resolved by implementing multithreading, where the LLM runs continuously in the background and remains sleep until triggered. The LLM is activated only when a valid plant name is pushed into its processing queue, allowing the vision pipeline to run smoothly without interruptions.

- The continuous triggering of the LLM when a plant remained in the camera frame. This resulted in the same plant name being sent repeatedly. To solve this problem, a temporal filtering mechanism was implemented. A plant must be detected consistently for at least ten consecutive frames before its name is pushed to the LLM queue. Once processed, the plant is marked as handled, ensuring that only one response is generated per plant instance.


## Future Improvements
- Replace terminal output with a graphical output screen
- Connect a mobile phone via API for remote usage
- Expand the dataset with more plant species




