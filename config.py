from configparser import ConfigParser
from argparse import ArgumentParser
from typing import Any

from os.path import exists, join, dirname, isfile


class SettingsBase(type):
    def __call__(cls, *args: tuple, **kwargs: dict) -> Any:
        if not hasattr(cls, 'instance'):
            cls.instance = super().__call__(*args, **kwargs)
        return cls.instance


class Settings(metaclass=SettingsBase):
    def __init__(self, default_config_file: str = 'config.ini'):
        if not exists(default_config_file) or not isfile(default_config_file):
            raise FileNotFoundError(f'Config file: [{default_config_file}] file is not found')

        if isfile(default_config_file):
            default_config_file = join(dirname(__file__), default_config_file)

        parser = ArgumentParser(
            description='Example script with argparse and config file.')

        # Add the -c/--config parameter
        parser.add_argument('-c', '--config',
                            default=default_config_file,
                            help='Specify the config file path')

        args = parser.parse_args()

        config = ConfigParser()
        config.read(args.config)

        # Access values from the configuration file
        self.__settings = {
            # Getting Main Path
            "path": dirname(__file__),

            # Getting Camera Connection Settings
            "username": config.get('Cameras', 'username').strip(),
            "password": config.get('Cameras', 'password').strip(),
            "ips": [ip.strip() for ip in config.get('Cameras', 'ips').split(',')],

            # Getting Foldernames Path To Save Images, Videos and Config Files
            "foldername": config.get('Saving', 'laf_image_foldername').strip(),
            "main_foldername": config.get('Saving', "main_foldername").strip(),
            "video_and_images_save_foldername": config.get('Saving', "video_and_images_save_foldername").strip(),
            "only_image_save_foldername": config.get('Saving', "only_image_save_foldername").strip(),
            "config_lines_foldername": config.get('Saving', 'config_lines_foldername').strip(),

            # Getting Saving Files Formats
            "line_config_file_format": config.get('Format', 'line_config_file_format').strip(),

            # Getting Acceptings
            "load_exists_config_file": config.get('Accept', 'load_exists_config_file').strip(),

            # Getting Using Model Settings
            "all_objects": config.get('Models', 'all_objects_model_path').strip(),
            "traffic_light": config.get('Models', "traffic_light_model_path").strip(),
        }

    def __str__(self) -> str:
        return str(self.__settings)

    def __repr__(self) -> str:
        return str(self.__settings)

    def __getitem__(self, __name: str) -> Any:
        return self.__settings.get(__name, None)


CONF = Settings()

if __name__ == "__main__":
    settings1 = Settings()
    settings2 = Settings()
    print(settings1)
    print(Settings)
