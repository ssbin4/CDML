# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 19:29:53 2016
@author: sakurai
"""

import os
import contextlib
import time
from six.moves.urllib import request
from progressbar import ProgressBar, Percentage, Bar, ETA, FileTransferSpeed
import fuel

try:
    fuel_root_path = fuel.config.config["data_path"]["yaml"]
except:
    fuel_root_path = ''
# New url for the dataset
base_url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/"
filename = "CUB200_2011/CUB_200_2011.tgz"
# These old URLs are no longer available
# base_url = "http://imagenet.stanford.edu/internal/car196/"
# filenames = ["car_ims.tgz", "cars_annos.mat"]
# urls = [base_url + f for f in filenames]

fuel_data_path = os.path.join(fuel_root_path, "cub200_2011")
os.mkdir(fuel_data_path)

url = base_url + filename

with contextlib.closing(request.urlopen(url)) as f:
    print(f.headers["content-length"])
    expected_filesize = int(f.headers["content-length"])
    print(expected_filesize)
time.sleep(5)

widgets = ['{}: '.format(filename), Percentage(), ' ', Bar(), ' ', ETA(),
           ' ', FileTransferSpeed()]
progress_bar = ProgressBar(widgets=widgets,
                           maxval=expected_filesize).start()


def reporthook(count, blockSize, totalSize):
    progress_bar.update(min(count*blockSize, totalSize))


filepath = os.path.join(fuel_data_path, filename)
request.urlretrieve(url, filepath, reporthook=reporthook)
progress_bar.finish()

downloaded_filesize = os.path.getsize(filepath)
assert expected_filesize == downloaded_filesize, " ".join((
    "expected file size is {}, but the actual size of the downloaded file",
    "is {}.")).format(expected_filesize, downloaded_filesize)