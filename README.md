## Assignment1

Comparison shows that only for really big numbers loops are slower.

Maximum number of threads and maximum number of blocks is 2147483647 - and number of threads/blocks must be 1 if max blocks/threads. In other case - allocation problem. Data on GPU can be seen from /usr/local/cuda-7.0/samples/1_Utilities/deviceQuery>./deviceQuery

Example for comparing times with maximum number of threads:

Time taken with gpu: 2e-05

Time taken with cpu: 18.5576
