from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
import time

# Model Name
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # or "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)

# Load inference model
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Look for GPU, move infernece model to GPU if avaliable otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def bench(prompt, max_length=50, runs=5):
    # '.to' to the device tensors (n-arrays) to put on.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warm up
    _ = model.generate(**inputs, max_length=max_length)

    times = []

    for _ in range(runs):
        start_time = time.time()
        with torch.no_grad(): # No gradient
            output_ids = model.generate(**inputs, max_length=max_length)
        times.append(time.time() - start_time)

        # decode
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    avg_time = sum(times) / runs
    print(f"Average Inference Time: {avg_time:.4f} sec")

    # tokens per second
    num_tokens = max_length
    throughput = num_tokens/avg_time
    print(f"Tokens per second: {throughput:.2f}")

    # memory or vram usage
    if torch.cuda.is_available():
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    print(response)

bench("What is hugging face?")
