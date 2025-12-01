# inference/engine_loader.py

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA Context 자동 생성
import numpy as np


class TRTInferenceEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)

        # TensorRT 엔진 로드 및 Execution Context 생성
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 바인딩 인덱스 추출
        self.input_idx = self.engine.get_binding_index("input")
        self.output_idx = self.engine.get_binding_index("output")

        # GPU 입력/출력 메모리 크기 계산
        input_shape = self.engine.get_binding_shape(self.input_idx)
        output_shape = self.engine.get_binding_shape(self.output_idx)
        self.input_size = int(np.prod(input_shape))
        self.output_size = int(np.prod(output_shape))

        # GPU 메모리 할당
        self.d_input = cuda.mem_alloc(self.input_size * np.float32().nbytes)
        self.d_output = cuda.mem_alloc(self.output_size * np.float32().nbytes)

        # Host 메모리(출력 수신용) 미리 할당
        self.h_output = np.empty(self.output_size, dtype=np.float32)

        # GPU 비동기 처리를 위한 CUDA Stream 생성
        self.stream = cuda.Stream()

    def infer(self, input_np):
        """TensorRT 모델 비동기 추론 실행 (PyCUDA 활용)"""

        # 입력 텐서를 float32 + 연속 메모리 형태로 보장
        input_np = np.asarray(input_np, dtype=np.float32)
        if not input_np.flags["C_CONTIGUOUS"]:
            input_np = np.ascontiguousarray(input_np)

        # Host → Device (비동기 전송)
        cuda.memcpy_htod_async(self.d_input, input_np, self.stream)

        # GPU에서 TensorRT 추론 비동기 실행
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle,
        )

        # Device → Host (비동기 수신)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)

        # 모든 비동기 작업 완료 대기
        self.stream.synchronize()

        return self.h_output.reshape(1, -1)  # (1, num_classes)
