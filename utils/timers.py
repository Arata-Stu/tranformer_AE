import torch
import time

class CudaTimer:
    """
    CUDAとCPUで計測可能なコンテキストマネージャ
    """
    def __init__(self, device):
        self.device = device
        if device == "cuda":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = None
            self.end_time = None

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device == "cuda":
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000  # 秒単位
        else:
            self.end_time = time.time()
            self.elapsed_time = self.end_time - self.start_time

    def get_time(self):
        return self.elapsed_time
