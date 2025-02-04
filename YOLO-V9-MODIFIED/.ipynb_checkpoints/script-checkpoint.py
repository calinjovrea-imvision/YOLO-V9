import torch
import multiprocessing as mp
from queue import Empty
from YOLO.yolo.model.yolo import create_model  # Assuming YOLOv9 model is defined in models.yolo

import sys
from pathlib import Path

from hydra import compose, initialize
from PIL import Image 

project_root = Path().resolve().parent
sys.path.append(str(project_root))

from YOLO.yolo.model.yolo import (
    AugmentationComposer,
    Config,
    PostProccess,
    create_converter,
    create_model,
    custom_logger,
    draw_bboxes,
)

# Load the YOLOv9 model
def load_model(model_path):
    model = create_model(model_path)  # Load YOLOv9 model
    model = model.cuda()  # Move model to GPU
    model.eval()  # Set model to evaluation mode
    return model

# Inference function for a single batch
def inference(model, input_batch):
    with torch.no_grad():  # Disable gradient computation
        input_batch = input_batch.cuda()  # Move input to GPU
        output = model(input_batch)  # Run inference
    return output

# Worker process for handling inference tasks
def worker(model, task_queue, result_queue):
    while True:
        try:
            # Get a batch of inputs from the task queue
            input_batch = task_queue.get(timeout=1)  # Timeout to avoid blocking indefinitely
            # Run inference
            output = inference(model, input_batch)
            # Put the result in the result queue
            result_queue.put(output)
        except Empty:
            break  # Exit if the queue is empty

# Main function to manage workload balancing
def main():
    # Load the YOLOv9 model
    model_path = 'path/to/yolov9.yaml'
    model = load_model(model_path)

    # Create task and result queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Generate example input data (replace with real data)
    input_data = [torch.randn(1, 3, 640, 640) for _ in range(16)]  # 16 example inputs

    # Dynamically batch inputs based on GPU memory
    batch_size = 4  # Adjust based on GPU memory
    for i in range(0, len(input_data), batch_size):
        batch = torch.cat(input_data[i:i + batch_size], dim=0)
        task_queue.put(batch)

    # Create worker processes
    num_workers = 2  # Number of parallel workers (adjust based on GPU capacity)
    workers = []
    for _ in range(num_workers):
        p = mp.Process(target=worker, args=(model, task_queue, result_queue))
        p.start()
        workers.append(p)

    # Wait for all workers to finish
    for p in workers:
        p.join()

    # Collect results from the result queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    print(f"Inference completed. Number of results: {len(results)}")

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Required for CUDA and multi-processing
    main()