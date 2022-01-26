'''
    Code adapted from https://gist.github.com/rusty1s/159eeff0d95e0786d220a164c6edd021
'''

import os
import os.path as osp
from six.moves import urllib
import errno
import tarfile

def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url, folder, log=True):
    print('Downloading', url)
    makedirs(folder)

    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2]
    path = osp.join(folder, filename)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


def extract_tar(path, folder, mode='r:gz', log=True):
    print('Extracting', path)
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


cwd = os.getcwd()
path = os.path.join( cwd, 'datasets/QM9/qm9' )
url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
      'datasets/gdb9.tar.gz'

file_path = download_url(url, path)
extract_tar(file_path, path, mode='r')