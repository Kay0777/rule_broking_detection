from typing import Callable, Any
from ipaddress import IPv4Address
from .classes import SaveModeEnum

class Validator:
    @staticmethod
    def ip_address_validator(func: Callable) -> Callable:
        def inner_func(*args, **kwargs):
            try:
                # Create an instance of IPv4Address
                IPv4Address(address=kwargs.get('ip'))
                return func(*args, **kwargs)
            except ValueError as e:
                # If ValueError is raised, the address is invalid
                raise ValueError('IP is not v4 Address!') from e
        return inner_func

    @staticmethod
    def save_validator(func: Callable) -> Callable:
        def inner_func(*args, **kwargs) -> Any:
            if len(args) == 3:
                _save_mode = args[2]
            else:
                _save_mode = kwargs.get('save_mode')

            if _save_mode is not None:
                # save_mode = getattr(args[0], 'save_mode')
                if isinstance(_save_mode, str) and _save_mode != '':
                    if _save_mode in {'npy', 'npz', 'jpg', 'jpeg'}:
                        save_mode = {
                            'npy': SaveModeEnum.npy,
                            'npz': SaveModeEnum.npz,
                            'jpg': SaveModeEnum.jpg,
                            'jpeg': SaveModeEnum.jpeg,
                        }[_save_mode]
                        setattr(args[0], 'save_mode', save_mode)
                elif isinstance(_save_mode, SaveModeEnum):
                    setattr(args[0], 'save_mode', _save_mode)
                else:
                    raise ValueError(
                        "Select any one of {'npy', 'jpeg', 'jpg', 'npz'} OR an enum from SaveModeEnum!")
            return func(*args, **kwargs)
        return inner_func