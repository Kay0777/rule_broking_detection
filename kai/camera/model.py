from multiprocessing.managers import SyncManager, Namespace, ValueProxy
from multiprocessing.shared_memory import SharedMemory
from threading import Event, Lock
from turbojpeg import TurboJPEG
import numpy as np
import os

from . import (
    Validator,
    SaveModeEnum,
    LafDataShmInfo,
    TensorDataShmInfo,
    SharedMemoryForNpNdArrayBase,
)
from typing import Union
from config import CONF


class SubControlCamera:
    __foldername = os.path.join(CONF['path'], CONF['foldername'])

    def __init__(self, foldername: str = __foldername) -> None:
        self.__foldername = self._create_folder(folder=foldername)

    @property
    def foldername(self) -> str:
        return self.__foldername

    @foldername.setter
    def foldername(self, value: str) -> None:
        self._create_folder(value)
        self.__foldername = value

    def _create_folder(self, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        return folder


class SubCamera(SubControlCamera):
    __ip: str
    __rtps_url: str

    def __init__(self, ip: str, username: str, password: str, save_mode: SaveModeEnum = SaveModeEnum.npy, *args, **kwargs) -> None:
        self.__create_ip_address(ip=ip)
        super().__init__(*args, **kwargs)

        self.__create_connect_rtsp_url(username=username, password=password)

        self.foldername: str = self._create_folder(
            folder=os.path.join(self.foldername, self.__ip))
        self.save_mode: SaveModeEnum = save_mode

    def __del__(self):
        try:
            del self.__ip
            del self.__rtps_url

            del self.foldername
            del self.save_mode
        except AttributeError as ex:
            print(ex)

    @property
    def ip(self) -> str:
        return self.__ip

    @property
    def url(self) -> str:
        return self.__rtps_url

    def __repr__(self) -> str:
        return self.__ip

    def __str__(self) -> str:
        return self.__ip

    def __hash__(self) -> int:
        return hash(self.__ip)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Camera):
            return self.__ip == __o.__ip
        return False

    @Validator.ip_address_validator
    def __create_ip_address(self, ip: str) -> None:
        self.__ip = ip

    def __create_connect_rtsp_url(self, username: str, password: str) -> None:
        self.__rtps_url = self.__class__.rtsp_url(username=username, password=password, ip=self.ip)

    @classmethod
    def rtsp_url(cls, username: str, password: str, ip: str) -> str:
        return f"rtsp://{username}:{password}@{ip}/cam/realmonitor?channel=1&subtype=0"


class Camera(SubCamera):
    def __init__(self, manager: SyncManager, ip: str, username: str, password: str, *args, **kwargs):
        super().__init__(ip=ip, username=username, password=password, *args, **kwargs)
        self.infos: Namespace = manager.Namespace()
        setattr(self.infos, 'lafInfo', None)
        setattr(self.infos, 'tensorsInfo', None)

        self.__isAlive: ValueProxy = manager.Value('b', True)
        self.__wait: Event = manager.Event()

        self.__lock: Lock = manager.Lock()
        self.__isBlocked: ValueProxy = manager.Value('b', False)

        self.__isDetectable: ValueProxy = manager.Value('b', False)

        self.__counter: ValueProxy = manager.Value('I', 1)
        self.__lastActiveQueue: ValueProxy = manager.Value('I', 1)

    def __del__(self) -> None:
        try:
            del self.__isAlive
            del self.__wait
            del self.__lock
            del self.__isBlocked
            del self.__counter
        except AttributeError:
            pass

    def wait(self):
        self.__wait.wait()

    def run(self) -> None:
        self.__wait.set()

    @property
    def isAlive(self) -> bool:
        return self.__isAlive.value

    @isAlive.setter
    def isAlive(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError('isAlive value setter error!')
        self.__isAlive.value = value

    # CAMERA COUNTER CONTROL PART
    # _____________________________________________________________________
    @property
    def counter(self) -> int:
        return self.__counter.value

    @counter.setter
    def counter(self, value: int) -> None:
        with self.__lock:
            self.__counter.value = value
    # _____________________________________________________________________

    # CAMERA BLOCK STATUS PART
    # _____________________________________________________________________
    @property
    def isBlocked(self) -> bool:
        return self.__isBlocked.value

    def block(self) -> None:
        with self.__lock:
            self.__isBlocked.value = True

    def unblock(self) -> None:
        with self.__lock:
            self.__isBlocked.value = False
    # _____________________________________________________________________

    # CAMERA IS DETECTABLE STATUS PART
    # _____________________________________________________________________
    @property
    def isDetectable(self) -> bool:
        return self.__isDetectable.value

    @isDetectable.setter
    def isDetectable(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError('Value type Error!')
        self.__isDetectable.value = value
    # _____________________________________________________________________

    # INITIALIZE SHARED MEMOERIES
    # GET LAST ACTIVE FRAME, UPDATE LAST ACTIVE FRAME
    # _____________________________________________________________________

    def init_laf_shared_memory(self, name: str, size: int) -> None:
        # Initialize Camera Last Active Frame Shared Memory Data
        SharedMemoryForNpNdArrayBase(name=name, size=size)

    def update_last_active_frame_data(self, newLafData: np.ndarray) -> None:
        name: str = f"Camera_LAF_SHm_{self.ip}"
        lafInfo: Union[None, LafDataShmInfo] = getattr(self.infos, 'lafInfo')
        if lafInfo is None:
            lafInfo = LafDataShmInfo(
                name=name,
                shape=newLafData.shape,
                dtype=newLafData.dtype)

            # Initialize Laf data Shared Memory
            self.init_laf_shared_memory(name=name, size=lafInfo.size)
            setattr(self.infos, 'lafInfo', lafInfo)

        SharedMemoryForNpNdArrayBase.update_shared_memory_data(
            shared_memory_name=name,
            new_data=newLafData,
            info=lafInfo)

        # Increment Laf data Queue
        self.__lastActiveQueue.value = self.__counter.value

    def last_active_frame_data(self) -> tuple[np.ndarray, int]:
        lafInfo: Union[None, LafDataShmInfo] = getattr(self.infos, 'lafInfo')
        if lafInfo is None:
            raise FileNotFoundError()

        lastActiveFrame = SharedMemoryForNpNdArrayBase.get_shared_memory_data(
            shared_memory_name=f"Camera_LAF_SHm_{self.ip}",
            info=lafInfo)
        return lastActiveFrame, self.__lastActiveQueue.value
    # _______________________________

    # _____________________ TENSOR SHARED MEMORY DATA _______________________ #
    # GET DETECTED TENSORS DATA, UPDATE TENSORS DATA
    def init_tensors_shared_memory(self, name: str, size: int) -> None:
        # Initialize Detected Tensors[Objects] Shared Memory Data From Last Active Frame
        SharedMemoryForNpNdArrayBase(name=name, size=size)

    def update_detected_tensors(self, boxes: np.ndarray) -> None:
        name: str = f"Camera_Tensors_SHm_{self.ip}"
        tensorsInfo: Union[None, TensorDataShmInfo] = getattr(self.infos, 'tensorsInfo')
        if tensorsInfo is None:
            tensorsInfo = TensorDataShmInfo(
                name=name,
                shape=boxes.shape,
                dtype=boxes.dtype)

            # Initialize Detected Tensors data Shared Memory
            self.init_tensors_shared_memory(name=name, size=tensorsInfo.size)
            setattr(self.infos, 'tensorsInfo', tensorsInfo)

        SharedMemoryForNpNdArrayBase.update_shared_memory_data(
            shared_memory_name=name,
            new_data=boxes,
            info=tensorsInfo)

    def detected_tensors(self) -> np.ndarray:
        tensorsInfo: Union[None, LafDataShmInfo] = getattr(self.infos, 'tensorsInfo')
        if tensorsInfo is None:
            raise FileNotFoundError()

        detectedTensors = SharedMemoryForNpNdArrayBase.get_shared_memory_data(
            shared_memory_name=f"Camera_Tensors_SHm_{self.ip}",
            info=tensorsInfo)
        return detectedTensors
    # _______________________________

    def cleaning_shared_memories(self):
        for shared_memory_name in (f"Camera_LAF_SHm_{self.ip}", f"Camera_Tensors_SHm_{self.ip}"):
            try:
                shm: SharedMemory = SharedMemory(name=shared_memory_name)
                shm.close()
                shm.unlink()

                print('File found and unliked!')
            except FileNotFoundError:
                print('Shared Memory is not found')
    # _____________________________________________________________________

    @Validator.save_validator
    def save(self, data: np.ndarray, save_mode: str | SaveModeEnum | None = None) -> None:
        if not isinstance(data, np.ndarray):
            raise ValueError('Saved data source error!')

        # if self.__shm is None:
        #     self.__create_a_shared_memory_data(data=data)

        filename = os.path.join(
            self.foldername,
            f'{self.counter}.{self.save_mode.name}'
        )
        if self.save_mode == SaveModeEnum.npy:
            np.save(file=filename, arr=data)
        elif self.save_mode == SaveModeEnum.npz:
            np.savez(file=filename, arr=data)
        else:
            try:
                jpeg = TurboJPEG()
                data = jpeg.encode(img_array=data, quality=85)  # type: ignore
            except Exception as ex:
                print('______________________________________')
                print(ex)
                print('______________________________________')
            finally:
                with open(file=filename, mode='wb') as file:
                    file.write(data)  # type: ignore
                    file.close()


if __name__ == "__main__":
    # with Manager() as manager:
    #     cams = [Camera(manager, ip, 1, 1) for ip in ['1.1.1.1', '1.1.1.2']]

    #     for cam in cams:
    #         print(cam.url)

    print("SaveModeEnum:", SaveModeEnum.npy.name)
    print(np.uint8, np.float64())
