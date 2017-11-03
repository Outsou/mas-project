import argparse
from scrap.timage_resave import resave_folder
import os


def make_pics(path, cm_name, shape=(1000, 1000)):
    files = os.listdir(path)
    dirs = [os.path.join(path, file) for file in files if os.path.isdir(os.path.join(path, file))]

    resave_folder(path, cm_name, shape)

    for dir in dirs:
        if dir[-6:] != 'no_gen':
            make_pics(dir, cm_name, shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recursively go through all subfolders and create images in path.")
    parser.add_argument('-l', metavar='folder', type=str, dest='run_folder', help='Folder for the runs.')
    args = parser.parse_args()
    folder = args.run_folder

    cm_name = None

    make_pics(folder, cm_name)
