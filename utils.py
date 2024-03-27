from argparse import ArgumentParser
from configparser import ConfigParser
from time import monotonic
import platform
import subprocess

from config import CONF


def cameraIPs() -> list[tuple[str, str, str]]:
    parser = ArgumentParser(
        description='Example script with argparse and config file.')

    # Add the -c/--config parameter
    parser.add_argument('-c', '--config', default='config.ini',
                        help='Specify the config file path')

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    # Access values from the configuration file
    username = config.get('Settings', 'username').strip()
    password = config.get('Settings', 'password').strip()
    ips = [ip.strip() for ip in config.get('Settings', 'ips').split(',')]

    return [(ip, username, password) for ip in ips]


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

def printt(message: str) -> None:
    if CONF['analysis_print'] == 'y':
        print(message)

def timeIt(camera: str, title: str, startTime: float) -> None:
    if CONF['analysis_print'] == 'y':
        print('Camera: {} || {}: {:.3f} ms'.format(camera, title, 1000 * (monotonic() - startTime)))
    


if __name__ == '__main__':
    cameraIPs()
