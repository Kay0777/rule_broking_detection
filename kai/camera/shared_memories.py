from multiprocessing.shared_memory import SharedMemory
from .classes import SharedMemoryForNpNdArrayInfo
import numpy as np

class SharedMemoryForNpNdArrayBase:
    def __init__(self, name: str, size: int) -> None:
        try:
            self.shm: SharedMemory = SharedMemory(
                name=name)
        except FileNotFoundError:
            self.shm: SharedMemory = SharedMemory(
                name=name,
                create=True,
                size=size)
    
    @staticmethod
    def update_shared_memory_data(shared_memory_name: str, new_data: np.ndarray, info: SharedMemoryForNpNdArrayInfo) -> None:
        shm: SharedMemory = SharedMemory(name=shared_memory_name)
        data = np.ndarray(shape=info.shape, dtype=info.dtype, buffer=shm.buf)
        data[:] = new_data[:]
        shm.close()

    @staticmethod
    def get_shared_memory_data(shared_memory_name: str, info: SharedMemoryForNpNdArrayInfo) -> np.ndarray:
        shm: SharedMemory = SharedMemory(name=shared_memory_name)
        data = np.ndarray(shape=info.shape, dtype=info.dtype, buffer=shm.buf)
        _data = data.copy()
        shm.close()
        return _data