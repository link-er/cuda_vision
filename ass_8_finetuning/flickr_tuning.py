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

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.plot(np.vstack([train_loss, scratch_train_loss]).T)
ax2 = fig.add_subplot(1,2,2)
ax2.plot(np.vstack([train_loss, scratch_train_loss]).clip(0, 4).T)
plt.savefig('/home/VI/stud/adilova/cuda_vision/ass_8_finetuning/training.png')

test_iters = 10
accuracy = 0
scratch_accuracy = 0
for it in arange(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
    scratch_solver.test_nets[0].forward()
    scratch_accuracy += scratch_solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters
scratch_accuracy /= test_iters
print 'Accuracy for fine-tuning:', accuracy
print 'Accuracy for training from scratch:', scratch_accuracy


