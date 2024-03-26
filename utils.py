from argparse import ArgumentParser
from configparser import ConfigParser
from time import monotonic


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


def timeIt(camera: str, title: str, startTime: float) -> None:
    print('Camera: {} || {}: {:.3f} ms'.format(camera, title, 1000 * (monotonic() - startTime)))


if __name__ == '__main__':
    cameraIPs()
