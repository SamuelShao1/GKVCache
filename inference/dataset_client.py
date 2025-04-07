import json
import random

def process(path):
    """
    Processes the input JSON file to extract and shuffle questions.
    Handles both:
    - UseCase4.json (with "Questions" and "ProcessedQuestion")
    - synthetic_data.json (array of {"question": ..., "answer": ...})
    """
    with open(path, 'r') as file:  
        data = json.load(file)
        
        if isinstance(data, list) and len(data) > 0 and "question" in data[0]:
            questions_list = [item["question"] for item in data if "question" in item]
        else:
            raise ValueError("Unsupported JSON structure")
        
        random.shuffle(questions_list)
        return questions_list
    
def distribute_prompts(file_path, num_clients):
    try:
        prompts = process(file_path)
        if not prompts:
            return []

        chunk_size = len(prompts) // num_clients
        distributed_prompts = [prompts[i * chunk_size:(i + 1) * chunk_size] for i in range(num_clients)]

        remaining_prompts = len(prompts) % num_clients
        for i in range(remaining_prompts):
            distributed_prompts[i].append(prompts[-(i + 1)])

        return distributed_prompts

    except Exception as e:
        print(f"An unexpected error occurred while distributing prompts: {e}")
        return []