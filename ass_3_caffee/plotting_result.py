
# coding: utf-8

# In[37]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

test_data = np.loadtxt('test_dataset1.txt')
test_data = np.transpose(np.delete(test_data, (0), axis=0))

result = np.loadtxt('test_classes1.txt')
result = [ int(x) for x in result ]

plt.scatter(test_data[0], test_data[1], c=result)


# In[ ]:




# In[ ]:



