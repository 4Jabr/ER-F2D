import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import time
import statistics
from dataloader import concatenate_subfolders
from utils.data_augmentation import CenterCrop

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTInference:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
        
        self.stream = cuda.Stream()
    
    def infer(self, image, events):
        np.copyto(self.inputs[0]['host'], image.ravel())
        np.copyto(self.inputs[1]['host'], events.ravel())
        
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)
        
        for inp in self.inputs:
            self.context.set_tensor_address(inp['name'], int(inp['device']))
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], int(out['device']))
        
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host'].reshape(1, 224, 224)

test_dataset = concatenate_subfolders(
    "Test/Town05",
    "SequenceSynchronizedFramesEventsDataset",
    "events/voxels", "depth/data", "rgb/data", 1,
    transform=CenterCrop(224),
    proba_pause_when_running=0.0, proba_pause_when_paused=0.0,
    step_size=1, clip_distance=80, every_x_rgb_frame=1,
    normalize='True', scale_factor=1, use_phased_arch="False",
    baseline="False", loss_composition=['image','event0'],
    reg_factor=3.70378, dataset_idx_flag=True, recurrency="False"
)

output_dir = "experiments/tensorrt_fp32/npy/depth"
os.makedirs(output_dir, exist_ok=True)

print("Loading TensorRT FP32 engine...")
model = TRTInference("tensorrt_fp32.trt")

print(f"Generating predictions for {len(test_dataset)} samples...")
time1 = []
memory_usage = []
start_total = time.time()

free_initial, total_mem = cuda.mem_get_info()

idx = 0
save_idx = 0
prev_dataset_idx = -1
sequence_idx = 0

while idx < len(test_dataset):
    item, dataset_idx = test_dataset[idx]
    
    if dataset_idx > prev_dataset_idx:
        sequence_idx = 0
    
    image = item[0]['image'].numpy().astype(np.float32)[np.newaxis, :]
    events = item[0]['events'].numpy().astype(np.float32)[np.newaxis, :]
    
    start = time.time()
    pred_depth = model.infer(image, events)
    latency = (time.time() - start) * 1000
    time1.append(latency)
    
    free_mem, _ = cuda.mem_get_info()
    current_used = (total_mem - free_mem) / (1024**2)
    memory_usage.append(current_used)
    
    if sequence_idx > 1:
        np.save(f"{output_dir}/{idx:010d}.npy", pred_depth)
        save_idx += 1
        
        if save_idx % 500 == 0:
            print(f"  Saved {save_idx}")
    
    sequence_idx += 1
    prev_dataset_idx = dataset_idx
    idx += 1

total_time = time.time() - start_total

if len(time1) > 10:
    warmup_skip = time1[10:]
    mean_latency = statistics.mean(warmup_skip)
    median_latency = statistics.median(warmup_skip)
    throughput = 1000.0 / mean_latency
    peak_memory = max(memory_usage)
    
    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS:")
    print(f"{'='*60}")
    print(f"Total samples:       {len(test_dataset)}")
    print(f"Saved predictions:   {save_idx}")
    print(f"Total time:          {total_time:.2f} seconds")
    print(f"Mean Latency:        {mean_latency:.2f} ms/sample")
    print(f"Median Latency:      {median_latency:.2f} ms/sample")
    print(f"Throughput:          {throughput:.2f} samples/second")
    print(f"Peak GPU Memory:     {peak_memory:.2f} MB")
    print(f"Speedup vs baseline: {58.16/mean_latency:.2f}x")
    print(f"{'='*60}\n")
