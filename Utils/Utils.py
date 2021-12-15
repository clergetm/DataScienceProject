import glob
import os


def delete_files(path):
	"""
	Delete files inside the given folder
	:param str path: the path to a folder to clear
	"""
	if os.path.isdir(path):
		for f in glob.glob(path + "/*"):
			os.remove(f)
			