import argparse
import os
import shutil


_AESTHETICS = ['benford', 'global_contrast_factor', 'fd_aesthetics', 'symm', 'entropy']
_CONVERSIONS = {'benford': 'benford',
                'entropy': 'entropy',
                'global_contrast_factor': 'gcf',
                'fd_aesthetics': 'frd',
                'symm': 'symm'}


def extract_aesthetics(file):
    aests = []
    agents = file.split('-')

    for agent in agents:
        for aest in _AESTHETICS:
            if aest in agent:
                aests.append(aest)
                break
    aests = list(map(lambda x: _CONVERSIONS[x], aests))
    return '-'.join(aests)

def copy_pics(path, output):
    files = os.listdir(path)
    dirs = [os.path.join(path, file) for file in files if os.path.isdir(os.path.join(path, file))]

    for file in files:
        if file[0] == 'f' and file[-4:] == '.txt':
            big_img = file[:-4] + '_None_1000x1000.png'
            small_img = 'bw' + file[1:-3] + 'png'
            aests = extract_aesthetics(file)

            i = 0
            name_found = False
            output_files = os.listdir(output)
            while not name_found:
                i += 1
                name = '{}{}.txt'.format(aests, i)
                if name not in output_files:
                    name_found = True
                    shutil.copyfile(os.path.join(path, file), os.path.join(output, name))
                    shutil.copyfile(os.path.join(path, big_img), os.path.join(output, name[:-3] + 'png'))

    for dir in dirs:
        if dir[-6:] != 'no_gen':
            copy_pics(dir, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add 1000x1000 images from source folder to gallery in output folder.")
    parser.add_argument('-l', metavar='folder', type=str, dest='run_folder', help='Folder for the runs.')
    parser.add_argument('-o', metavar='folder', type=str, dest='output_folder', help='Folder for the output.')
    args = parser.parse_args()
    folder = args.run_folder
    output_folder = args.output_folder

    copy_pics(folder, output_folder)