## Assignment1

Comparison shows that only for really big numbers loops are slower.

Maximum number of threads and maximum number of blocks is 2147483647 - and number of threads/blocks must be 1 if max blocks/threads. In other case - allocation problem. Data on GPU can be seen from /usr/local/cuda-7.0/samples/1_Utilities/deviceQuery>./deviceQuery

Example for comparing times with maximum number of threads:

Time taken with gpu: 2e-05

Time taken with cpu: 18.5576

## Assignment2

Contains ipython notebook (old version and new version, old for university PC) that contains several sample data generators.

Cpp project reads matrix from the text file generated in python (with some additions for usability) and multiplies it by 2 in CUDA.

## Assignment3

Classification with help of caffee blobs of simple multivariant gaussian classes.

Works with 100% correctness.

## Assignment4

For using previous implementation of K-means with MNIST data it is needed to reshape input matrices of 28x28 to vectors (blob->Reshape command). The result was 0.0309 error rate.

With argmax layer program works definitely faster with the same results of error.

## Assignment5

Using softmax technique for classifying MNIST dataset with 10 classes for classification. When learning on GPU time is (for 1000 iterations and learning rate 0.1) 51.3484s. When learning on CPU (same parameters) - 748.319s. Error development is displayed with help of python.

Score with learning rate 0.1 was 9176 out of 10000. With learning rate 0.01 I got smaller errors while learning and score was about the same, but sometimes it was giving very low scores, while with 0.1 rate it was more stable. With 0.001 the smallest loss while learning fall till almost 2 and scores were about 9000 as well. With 0.9 scores again were about 9000, but errors did not fall below 6. With learning rate 1.5 the picture was the same.

## Assignment6

Learning MNIST data with LeNet - achieved accuracy is about 9700 from 10000 (learning rate 0.1, 921 iterations with 64 batch size)


