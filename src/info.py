import json
import ollama
import queue
import threading


with open("information.json", "r") as f:
    homeplants = json.load(f)

plant_q = queue.Queue()

#preventing repeated LLM calls for same plant
done_plants = set()



def generate_info(plant_name):
    """takes plant name 
    then print infos"""

    key = plant_name.lower().strip()
    
    if key not in homeplants:
        print(f"No data for {plant_name}")
        return

    data = homeplants[key]


    prompt = f"""
You are a plant expert.

Plant detected: {plant_name}

ONLY use this data:
{json.dumps(data, indent=2)}

Your task is to explain simply in separate paragraphs:
- Scientific + common name
- Care instructions like watering and light
- How to grow
- Toxicity warning if needed

Provide the information in a friendly way as you are speaking with 6 years old child.
"""

    response = ollama.generate(
        model="qwen3:0.6b-q8_0",
        prompt=prompt
    )

    #response = ollama.generate(
    #    model="phi3:latest",
    #    prompt=prompt
    #)

    print("\n" + "-" * 30)
    print(f"WOW it is {plant_name}")
    print(response["response"])
    print("-" * 30 + "\n")


def llm_exec():
    while True:
        plant_name = plant_q.get()
        
        #to run LLM only once per plant
        if plant_name in done_plants:
            plant_q.task_done()
            continue

        done_plants.add(plant_name)
        generate_info(plant_name)
        
        #tell queue it is done
        plant_q.task_done()


th = threading.Thread(target=llm_exec, daemon=True)
th.start()