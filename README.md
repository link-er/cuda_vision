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

Learning MNIST data with LeNet - achieved accuracy is about 9700 from 10000 (learning rate 0.1, 921 iterations with 64 batch size). When learning was held for 5000 iterations the result was about 9909 out of 10000.

## Assignment7

Results of working network with 5000 iterations were:

Loss: 0.0341908

Test score: 9892 out of 10000

So it is about the same as were in previous net. 

## Assignment8

For finetuning CIFAR10 resized to 256x256 dataset the accuracy result, after 200 iterations, was 0.742000001669

For learning from scratch the same model and same solver gave result of 0.332000002265 accuracy

## Assignment9

Nearest Neighbor with 40000 train images and 10000 test images gives Error rate equal to 0.6583. Learning time with GPU: 38.701. Learning time with CPU: 1175.53

Logistic Classifier with all 50000 train images and 10000 test images gave score 2822 out of 10000. Learning time: 163.399

Logistic Classifier on CPU - Learning time: 2563.72; Test score: 2963 out of 10000

CNN Training time with GPU on 500 iterations: 96.0244548321

CNN Training time with CPU on 500 iterations: 1825.20337296
