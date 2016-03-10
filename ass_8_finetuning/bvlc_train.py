path_caffe = '/home/VI/stud/adilova/caffe-master/'
import sys
sys.path.insert(0, path_caffe + 'python')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lmdb
from pylab import *
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/home/VI/stud/adilova/cuda_vision/ass_8_finetuning/bvlc_solver.prototxt')

niter = 200
train_loss = np.zeros(niter)

for it in range(niter):
    solver.step(1)  # SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data
    if it % 10 == 0:
        print 'iter %d, loss=%f' % (it, train_loss[it])
