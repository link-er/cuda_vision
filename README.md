## Assignment1

Comparison shows that only for really big numbers loops are slower.

Maximum number of threads greater than 1024, but maximum number of blocks is 2147483647 - and number of threads must be 1 if max blocks. In other case - allocation problem. Data on GPU can be seen from /usr/local/cuda-7.0/samples/1_Utilities/deviceQuery>./deviceQuery