import os
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Check collaboration run statuses in given  base folder.")
	parser.add_argument('-l', metavar='folder', type=str, dest='run_folder', help='Folder for the runs.')

	args = parser.parse_args()
	folder = args.run_folder
	d = os.listdir(folder)
	for e in d:
		run_path = os.path.join(folder, e)
		if os.path.isdir(run_path):
			with open(os.path.join(run_path, 'rinfo.txt'), 'r') as f:
				content = f.readlines()
				print("{}: {}".format(run_path, content[-1].strip()))

