import os
import sys
import tarfile

gen_fn_dir = os.path.abspath('..') + '/shared_scripts'
sys.path.append(gen_fn_dir)
import general_file_fns as gff


def extract_file(path):
    dir_list = os.listdir(path)
    for fname in dir_list:
        if fname.endswith("tar.gz"):
            print(path + fname)
            tar = tarfile.open(path+fname, "r:gz")
            tar.extractall(path=path)
            tar.close()
            print("Remove " + fname)
            os.remove(path+fname)


gen_params = gff.load_pickle_file('../general_params/general_params.p')
data_path = gen_params['raw_data_dir'] + '/'
extract_file(data_path)
