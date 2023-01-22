#!/usr/bin/env python
# coding: utf-8

# A little pre-requisites about array and reshape

# ![image.png](attachment:image.png)

# In[50]:


from PIL import Image
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import random


# ### Read image and store it as an 3 dimensional array

# In[4]:


img = Image.open(f"photo.jpeg")
print(img)


# In digital imaging, a pixel(or picture element) is the smallest item of information in an image. 

# In[45]:


img_arr = np.array(img, dtype='int32')
print('The shape is {}, meaning {} x {} pixels (picture elements) with {}-coded RGB.\n'.format(str(img_arr.shape),\
                                                                                str(img_arr.shape[0]),\
                                                                                str(img_arr.shape[1]),\
                                                                                str(img_arr.shape[2]))) # 3 dimensional image array
print('img_arr\'s 1st pixel:','\n',img_arr[0][0])
print('')
print('img_arr\'s last pixel:','\n',img_arr[-1][-1])


# ### Use the 3 dimensional array to display the image

# In[47]:


arr = img_arr.astype(dtype='uint8')
print(arr[0])


# `Image.fromarray`
# 
# Creates an image memory from an object exporting the array interface (using the buffer protocol).
# 
# `matplotlib.pyplot.imshow(X)`
# 
# Display data as an image
# 
# `X`: array-like or PIL image. The image data. Supported array shapes are:
# 
# - (M, N): an image with scalar data. The values are mapped to colors using normalization and a colormap. See parameters norm, cmap, vmin, vmax.
# - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
# - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int), i.e. including transparency.
# - The first two dimensions (M, N) define the rows and columns of the image. Out-of-range RGB(A) values are clipped.

# In[14]:


img = Image.fromarray(arr, mode='RGB') # (3x8-bit pixels, true color)

imshow(np.asarray(img))


# In[17]:


try:
    Image.fromarray(arr, mode='CMYK') # (4x8-bit pixels, color separation)
except ValueError:
    print('Failed - CMYK is for 4 dimensional array')


# ### reshape the matrix into "img_reshaped" by transforming "img_arr" from a "3-D" matrix to a flattened "2-D" matrix which has 3 columns corresponding to the RGB values for each pixel

# In[21]:


img_reshaped = img_arr.reshape([img_arr.shape[0]*img_arr.shape[1], 3])
print(img_reshaped.shape) # 2048*1548 = 3170304


# In[49]:


print('img_reshaped\'s 1st pixel:','\n',img_reshaped[0])
print('')
print('img_reshaped\'s last pixel:','\n',img_reshaped[-1])


# The two examples above are same as the orginal 3-dimensional array

# ### create k-means function that groups a large volume of inputs into a small number of clusters

# In[60]:


print('# Low and high are both specified.')
print(np.random.randint(low=2, high=5, size=10)) 
print('\n# If high is None (the default), then results are from [0, low).')
print(np.random.randint(2, size=10)) 


# In[73]:


# initialize k centers randomly
def initialize_centers(X, k):
    return X[np.random.randint(len(X), size=k): ] # slicing to returns centers as a (k x d) numpy array.


# In[74]:


# computes a distance matrix, storing the squared distance from point x to center j. 
def compute_squared_distance(X, centers):
    m = len(X) 
    k = len(centers) 
    
    S = np.empty((m, k)) 
    
    for i in range(m): 
        S[i,:] = np.linalg.norm(X[i,:] - centers, ord = 2, axis = 1) ** 2
        
    return S


# In[75]:


def compute_squared_distance_v2(X, centers):
    ans = []
    for i in range(len(centers)):
        ans.append(np.sum((X - centers[i])**2, axis = 1))
    
    # check out the concept under the pre-requisite section
    a = np.array(ans).reshape(centers.shape[1], X.shape[0]) # first horizontal, then vertical
    b = a.T # first vertical, then horizontal
    return b


# In[83]:


# Given a clustering (i.e., a set of points and assignment of labels), compute the center of each cluster.
def find_centers(X, labels):
    # X[:m, :d] == m points, each of dimension d
    m, d = X.shape
    
    # this time to find one more cluster than previously
    k = int(max(labels) + 1)
    
    assert m == len(labels) # the number of points in X must equal the number of assigned labels
    assert (min(labels) >= 0) # labels must start from 0

    # create a blank array for original plus one more cluster
    centers = np.empty((k, d)) 
    for i in range(k):
        # Compute the new center of cluster j, the mean of all points within the same cluster i
        centers[i, :] = np.mean(X[labels == i, :], axis = 0)
    return centers    
    


# In[77]:


# use the squared distance matrix to find each point's minimum squared distance and assign a "cluster label" to it
def assign_cluster_labels(S):
    return np.argmin(S, axis = 1)


# In[78]:


# Given the squared distances, return the within-cluster sum of squares.
def calculate_wcss(S):
    return np.sum(np.amin(S, axis=1)) # Return the minimum of an array or minimum along an axis.


# In[80]:


def kmeans(X, k,
           starting_centers=None,
           max_steps=np.inf):
    
    # initialize k centers by choice or randomly
    if starting_centers is None:
        centers = initialize_centers(X, k)
    else:
        centers = starting_centers
        
    converged = False
    # by default, give every point the label of 0, to update later
    labels = np.zeros(len(X))
    wcss = 0
    
    i = 1
    # keep iterating as long as it's not converged and hasn't hit the maximum iteration allowance
    while (not converged) and (i <= max_steps):
        old_centers = centers
        old_wcss = wcss
        
        S = compute_squared_distance(X, old_centers)
        labels = assign_cluster_labels(S)
        
        centers = find_centers(X, labels)
        wcss = calculate_wcss(S)
        print ("iteration", i, "WCSS = ", wcss)
        i += 1
        
        # break the loop if new iteration is not better (a.k.a wcss is not less)
        if old_wcss == wcss:
            converged = True
            
    return labels


# ### Apply the k-means function to divide the image in 3 clusters. The result would be a vector named labels, which assigns the label to each pixel.
# 

# In[84]:


clusters = kmeans(img_reshaped, k=3, starting_centers=img_reshaped[[0, 0], :],max_steps=np.inf)


# In[85]:


clusters


# In[ ]:




