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
import time

caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('/home/VI/stud/adilova/cuda_vision/ass_9_cifar/cnn/cifar_classifier_solver.prototxt')

# each output is (batch size, feature dim, spatial dim)
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]

# just print the weight sizes (we'll omit the biases)
print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (there can be more than one)

solver.step(1)
plt.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 8, 5, 5).transpose(0, 2, 1, 3).reshape(4*8, 5*5), cmap='gray'); plt.axis('off')
plt.savefig('/home/VI/stud/adilova/cuda_vision/ass_9_cifar/cnn/conv1_1.png')

plt.imshow(solver.net.params['conv2'][0].diff[:, 0].reshape(4, 8, 5, 5).transpose(0, 2, 1, 3).reshape(4*8, 5*5), cmap='gray'); plt.axis('off')
plt.savefig('/home/VI/stud/adilova/cuda_vision/ass_9_cifar/cnn/conv2_1.png')

plt.imshow(solver.net.params['conv3'][0].diff[:, 0].reshape(8, 8, 5, 5).transpose(0, 2, 1, 3).reshape(8*8, 5*5), cmap='gray'); plt.axis('off')
plt.savefig('/home/VI/stud/adilova/cuda_vision/ass_9_cifar/cnn/conv3_1.png')

plt.close('all')

#time
niter = 500
test_interval = 25
# losses will also be stored in the log
train_loss = np.zeros(niter)
test_acc = np.zeros(int(np.ceil(niter / test_interval)))
output = np.zeros((niter, 8, 10))

start = time.time()
# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data

    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['ip1'].data[:8]

    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(100):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['ip1'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

end = time.time()
print "Training time: " + str(end - start)

plt.imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 8, 5, 5).transpose(0, 2, 1, 3).reshape(4*8, 5*5), cmap='gray'); plt.axis('off')
plt.savefig('/home/VI/stud/adilova/cuda_vision/ass_9_cifar/cnn/conv1_2.png')

plt.imshow(solver.net.params['conv2'][0].diff[:, 0].reshape(4, 8, 5, 5).transpose(0, 2, 1, 3).reshape(4*8, 5*5), cmap='gray'); plt.axis('off')
plt.savefig('/home/VI/stud/adilova/cuda_vision/ass_9_cifar/cnn/conv2_2.png')

plt.imshow(solver.net.params['conv3'][0].diff[:, 0].reshape(8, 8, 5, 5).transpose(0, 2, 1, 3).reshape(8*8, 5*5), cmap='gray'); plt.axis('off')
plt.savefig('/home/VI/stud/adilova/cuda_vision/ass_9_cifar/cnn/conv3_2.png')

plt.close('all')

_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(np.arange(niter), train_loss)
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.savefig('/home/VI/stud/adilova/cuda_vision/ass_9_cifar/cnn/test_accuracy.png')
