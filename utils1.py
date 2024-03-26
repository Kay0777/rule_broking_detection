from enum import Enum
from dataclasses import dataclass
from os import path as OSPath, mkdir
import argparse
import configparser
import platform
import subprocess
from models import CameraIP


def create_folders(foldername: str, ips: list[str]) -> list[str]:
    _path = OSPath.join(OSPath.dirname(__file__), foldername)

    if not OSPath.exists(_path):
        mkdir(_path)

    folders = []
    for ip in ips:
        camera_foldername = OSPath.join(_path, ip)
        if not OSPath.exists(camera_foldername):
            mkdir(camera_foldername)
        folders.append(camera_foldername)
    return folders


def camera_ips() -> list[CameraIP]:
    parser = argparse.ArgumentParser(
        description='Example script with argparse and config file.')

    # Add the -c/--config parameter
    parser.add_argument('-c', '--config', default='config.ini',
                        help='Specify the config file path')

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    # Access values from the configuration file
    username = config.get('Cameras', 'username').strip()
    password = config.get('Cameras', 'password').strip()
    ips = [ip.strip() for ip in config.get('Cameras', 'ips').split(',')]

    # Get foldername
    foldername = config.get('Saving', 'foldername').strip()
    folders = create_folders(foldername=foldername, ips=ips)

    return [
        CameraIP(
            ip=ip,  # type: ignore
            url="rtsp://{}:{}@{}/cam/realmonitor?channel=1&subtype=0".format(
                username, password, ip),
            folder=folder
        ) for ip, folder in zip(ips, folders)
    ]


def ping(host: str) -> bool:
    """
    Returns True if host (str) responds to a ping request.
    Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
    """

    # Option for the number of packets as a function of
    param = '-n' if platform.system().lower() == 'windows' else '-c'

    # Building the command. Ex: "ping -c 1 google.com"
    command = ['ping', param, '1', host]

    return subprocess.call(command) == 0
