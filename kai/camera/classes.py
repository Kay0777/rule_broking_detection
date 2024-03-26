from dataclasses import dataclass
from numpy import dtype, prod

from enum import Enum

class SaveModeEnum(Enum):
    npy = 'npy'
    npz = 'npz'
    jpg = 'jpg'
    jpeg = 'jpeg'

@dataclass
class SharedMemoryForNpNdArrayInfo:
    name: str
    shape: tuple
    dtype: dtype
    
    def __post_init__(self):
        self.size = int(prod(self.shape) * dtype(self.dtype).itemsize)


@dataclass
class LafDataShmInfo(SharedMemoryForNpNdArrayInfo):
    pass

@dataclass
class TensorDataShmInfo(SharedMemoryForNpNdArrayInfo):
    pass