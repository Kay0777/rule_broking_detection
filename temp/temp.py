import os


def read_data_from_labels_asilbek(file: str) -> None:
    with open(file=file, mode='r') as f:
        data = [dt.split('\n')[0] for dt in f.readlines()]
        f.close()
        info = []
        for i in data:
            if i[0] == '0':
                info.append(f'7 {i[1:].strip()}')
                continue
            info.append(f'5 {i[1:].strip()}')
        return info


def read_data_from_images_azimjon(file: str) -> None:
    with open(file=file, mode='r') as f:
        data = [dt.split('\n')[0] for dt in f.readlines() if dt.split('\n')[0][0] not in ('7, 5')]
        f.close()
        return data


def write_merged_data(file: str, data: list) -> None:
    with open(file=file, mode='w') as f:
        f.writelines(data)
        f.close()

# 0 => 7
# 1 => 5


pwd = os.path.dirname(__file__)
folder = os.path.join(pwd, 'labels')
asilbek_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('txt')]
for asilbek_file in asilbek_files:
    info1 = read_data_from_labels_asilbek(file=asilbek_file)

    azimjon_file: str = asilbek_file.replace('labels', 'images')
    info2 = []
    if os.path.exists(azimjon_file):
        info2 = read_data_from_images_azimjon(file=azimjon_file)
    data = [f'{info}\n' for info in info1 + info2]

    file = asilbek_file.replace('labels', 'changed')
    write_merged_data(file=file, data=data)
