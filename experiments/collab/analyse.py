from utils.stats import analyze_collab_gp_runs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform statistical analysis on runs in given folder.")
    parser.add_argument('-l', metavar='folder', type=str, dest='run_folder', help='Folder for the runs.')

    args = parser.parse_args()
    folder = args.run_folder
    analyze_collab_gp_runs(folder)
