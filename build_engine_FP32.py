import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

print("Building TensorRT FP32 engine...")
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open("model.onnx", 'rb') as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
# No FP16 flag = FP32

profile = builder.create_optimization_profile()
profile.set_shape("image", (1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224))
profile.set_shape("events", (1, 5, 224, 224), (1, 5, 224, 224), (1, 5, 224, 224))
config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)
with open("tensorrt_fp32.trt", 'wb') as f:
    f.write(serialized_engine)

print("FP32 engine saved!")
