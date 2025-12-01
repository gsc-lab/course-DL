import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def main():
    with open("model.trt", "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    input_idx = engine.get_binding_index("input")
    output_idx = engine.get_binding_index("logits")

    input_shape = engine.get_binding_shape(input_idx)
    output_shape = engine.get_binding_shape(output_idx)

    print("Input shape :", input_shape)
    print("Output shape:", output_shape)

if __name__ == "__main__":
    main()
    # TensorRT 객체들(runtime, engine, context)은 여기서 스코프 밖으로 나가면서 먼저 파괴
    import gc
    gc.collect()  # 파괴 시점 조금 더 당겨주는 용도 (필수는 아님)
