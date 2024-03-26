from openvino.runtime import Core, CompiledModel, Model, AsyncInferQueue
from openvino.runtime.ie_api import InferRequest
# from openvino
import numpy as np

from typing import Union, Any
from .classes import ModelType, Device

class TensorModel:
    _instances: dict = {}

    def __new__(cls, *args, **kwargs):
        modelType: ModelType = kwargs['modelType'] if 'modelType' in kwargs else args[0]

        if modelType.name not in cls._instances:
            cls._instances[modelType.name] = super(TensorModel, cls).__new__(cls)
            cls._instances[modelType.name].modelType = modelType
        return cls._instances[modelType.name]

    def __init__(self, modelType: ModelType=ModelType.ALL, device: Device=Device.CPU) -> None:
        if not isinstance(modelType, ModelType):
            raise TypeError('Model type is error! Unknown model type!')

        if not isinstance(device, Device):
            raise TypeError('Device type is error! Unknown devide type!')

        self.__taskQueueSize: int = 1

        # Create an OpenVino Core
        self.openVinoCore: Core = Core()

        # Initialize Model Type
        self.__modelType: ModelType = modelType

        # Loading Model to Core
        self.model: Model = self.openVinoCore.read_model(model=self.__modelType.type)
        
        # Create an Compiled Model from Core
        self.__compiledModel: CompiledModel = self.openVinoCore.compile_model(
            model=self.model,
            device_name=device.type)
        
        # Create an Infer Requester
        self.__asyncInferQueue: AsyncInferQueue = AsyncInferQueue(
            model=self.__compiledModel,
            jobs=self.__taskQueueSize)
        self.__asyncInferQueue.set_callback(self.__get_results)
    
    def __repr__(self) -> str:
        return f"[TensorModel: {self.__modelType}]"

    def __str__(self) -> str:
        return f"[TensorModel: {self.__modelType}]"
    
    def __get_results(self, infer_request: InferRequest, _: Any) -> None:
        # Check if the inference is complete and handle the results
        self.__results = infer_request.results[self.__compiledModel.output(0)]

    @property
    def isAsyncInferQueueReady(self):
        return self.__asyncInferQueue.is_ready()

    def detect(self, inTenSorData: np.ndarray, name: Union[int, str]=0) -> np.ndarray:
        # Start Asynchronously detection
        while not self.isAsyncInferQueueReady:
            pass
        
        # Async Start Analyze by Model
        self.__asyncInferQueue.start_async(inputs={name: inTenSorData})
    
        # Wait till task is done For GIL is not blocked!
        self.__asyncInferQueue.wait_all()
        
        # Return done results list [Tensors]
        return self.__results


def main():
    model1 = TensorModel(modelType=ModelType.ALL)
    model2 = TensorModel(modelType=ModelType.ALL)

    print(id(model1))
    print(id(model2))
    print(id(model1) is id(model2))

if __name__ == "__main__":
    main()