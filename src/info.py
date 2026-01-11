import json
import ollama

with open("information.json", "r") as f:
    PLANTS = json.load(f)

def generate_info(plant_name):
    key = plant_name.lower().strip()

    if key not in PLANTS:
        print("No info")
        return
    
    data = PLANTS[key]

    
    
    
    prompt = f"""
You are a plant expert. A vision system has detected a plant: **{plant_name}**.

IMPORTANT RULES:
- you MUST ONLY use the information provided in the JSON below
- Do NOT add new facts, or assumptions
- MAKE it simple 
- you are NOT allowed to invent anything oustide JSON

Here is the structured data from the knowledge base:
{json.dumps(data, indent=2)}

Your task:
1. Give the scientific name and common name people usually use.
2. Give care instructions: soil, watering.
3. Explain briefly how to grow or maintain the plant.
4. If the plant is toxic for pets include a warning.

Make the explanation clear and simple. make every task in seperate paragraph under a tital. Do not use external knowladge excpet if it was basic and sure of it
    """

    response = ollama.generate(
        model="phi3",
        prompt=prompt
    )

    return response["response"]

if __name__ == "__main__":
    print(generate_info)