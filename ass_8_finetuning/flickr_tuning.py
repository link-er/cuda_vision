path_caffe = '/home/VI/stud/adilova/caffe-master/'
import sys
sys.path.insert(0, path_caffe + 'python')

import numpy as np
import matplotlib.pyplot as plt
import lmdb
from pylab import *
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(path_caffe+'models/finetune_flickr_style/solver.prototxt')
solver.net.copy_from(path_caffe+'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

scratch_solver = caffe.SGDSolver(path_caffe+'models/finetune_flickr_style/solver.prototxt')

niter = 200
train_loss = np.zeros(niter)
scratch_train_loss = np.zeros(niter)

for it in range(niter):
    solver.step(1)  # SGD by Caffe
    scratch_solver.step(1)
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
    if it % 10 == 0:
        print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])

