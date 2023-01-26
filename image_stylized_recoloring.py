#!/usr/bin/env python
# coding: utf-8

# A little pre-requisites about array and reshape

# ![image.png](attachment:image.png)

# In[68]:


from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import random


# ### Read image and store it as an 3 dimensional array

# In[69]:


img = Image.open(f"photo.jpeg")
print(img)


# In digital imaging, a pixel(or picture element) is the smallest item of information in an image. 

# In[70]:


img_arr = np.array(img, dtype='int32')
print('The shape is {}, meaning {} x {} pixels (picture elements) with {}-coded RGB.\n'.format(str(img_arr.shape),\
                                                                                str(img_arr.shape[0]),\
                                                                                str(img_arr.shape[1]),\
                                                                                str(img_arr.shape[2]))) # 3 dimensional image array
print('img_arr\'s 1st pixel:','\n',img_arr[0][0])
print('')
print('img_arr\'s last pixel:','\n',img_arr[-1][-1])


# ### Use the 3 dimensional array to display the image

# In[71]:


arr = img_arr.astype(dtype='uint8')
print(type(arr))
print('\nShape:', arr.shape, '\n')
print(arr[0])


# `matplotlib.pyplot.imshow(X)`
# 
# Display data as an *inline* image
# 
# `X`: array-like or PIL image. The image data. Supported array shapes are:
# 
# - (M, N): an image with scalar data. The values are mapped to colors using normalization and a colormap. See parameters norm, cmap, vmin, vmax.
# - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
# - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int), i.e. including transparency.
# - The first two dimensions (M, N) define the rows and columns of the image. Out-of-range RGB(A) values are clipped.

# In[104]:


imshow(arr)
plt.title('Original')


# `Image.fromarray`
# 
# Creates an image memory from an object exporting the array interface (using the buffer protocol).
# 
# 

# In[73]:


img = Image.fromarray(arr, mode='RGB') # (3x8-bit pixels, true color)
print(type(img))
# img


# In[74]:


try:
    Image.fromarray(arr, mode='CMYK') # (4x8-bit pixels, color separation)
except ValueError:
    print('Failed - CMYK is for 4 dimensional array')


# ### Reshape the matrix by transforming "img_arr" from a "3-D" matrix to a flattened "2-D" matrix  
# It has 3 columns corresponding to the RGB values for each pixel

# In[75]:


img_reshaped = img_arr.reshape([img_arr.shape[0]*img_arr.shape[1], 3])
print(img_reshaped.shape) # 2048*1548 = 3170304


# In[76]:


print('img_reshaped\'s 1st pixel:','\n',img_reshaped[0])
print('')
print('img_reshaped\'s last pixel:','\n',img_reshaped[-1])


# The two examples above are same as the orginal 3-dimensional array

# ### Create k-means function that groups a large volume of inputs into a small number of clusters

# In[77]:


print('# Low and high are both specified.')
print(np.random.randint(low=2, high=5, size=10)) 
print('\n# If high is None (the default), then results are from [0, low).')
print(np.random.randint(2, size=10)) 


# In[78]:


# initialize k centers randomly
def initialize_centers(X, k):
    return X[np.random.randint(len(X), size=k), : ] # slicing to returns centers as a (k x d) numpy array.


# In[79]:


# computes a distance matrix, storing the squared distance from point x to center j. 
def compute_squared_distance(X, centers):
    m = len(X) 
    k = len(centers) 
    
    S = np.empty((m, k)) 
    
    for i in range(m): 
        S[i,:] = np.linalg.norm(X[i,:] - centers, ord = 2, axis = 1) ** 2
        
    return S


# In[80]:


def compute_squared_distance_v2(X, centers):
    ans = []
    for i in range(len(centers)):
        ans.append(np.sum((X - centers[i])**2, axis = 1))
    
    # check out the concept under the pre-requisite section
    a = np.array(ans).reshape(centers.shape[1], X.shape[0]) # first horizontal, then vertical
    b = a.T # first vertical, then horizontal
    return b


# In[81]:


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
    


# In[82]:


# use the squared distance matrix to find each point's minimum squared distance and assign a "cluster label" to it
def assign_cluster_labels(S):
    return np.argmin(S, axis = 1)


# In[83]:


# Given the squared distances, return the within-cluster sum of squares.
def calculate_wcss(S):
    return np.sum(np.amin(S, axis=1)) # Return the minimum of an array or minimum along an axis.


# In[84]:


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


# ### Apply the k-means function to divide the image in 3 clusters. 
# The result would be a vector named labels, which assigns the label to each pixel.
# 

# In[85]:


clusters = kmeans(img_reshaped, k=3, starting_centers=img_reshaped[[0, 0], :],max_steps=np.inf)


# In[86]:


clusters


# In[87]:


set(clusters)


# Specifying 1 point by [0,0] was a wrong setup, it then returned only 1 cluster (a.k.a label 0).

# In[88]:


clusters2 = kmeans(img_reshaped, k=3, starting_centers=None,max_steps=np.inf)


# In[89]:


clusters2


# In[90]:


set(clusters2)


# It looks more reasonable now - three clusters (three labels).

# ### Calculate the mean of each cluster 

# In[91]:


centers = {}
centers[0] = np.mean(img_reshaped[clusters2==0], axis = 0)
centers[1] = np.mean(img_reshaped[clusters2==1], axis = 0)
centers[2] = np.mean(img_reshaped[clusters2==2], axis = 0)
centers


# ### Generate a matrix of the same dimensions, where each pixel is replaced by the cluster center to which it belongs.

# In[92]:


img_clustered = np.array([centers[i] for i in clusters2])
img_clustered


# ### Display the clustered image

# In[93]:


arr_clustered = img_clustered.astype(dtype='uint8')
print(type(arr_clustered))
print()
print(arr_clustered)


# In[94]:


imshow(arr_clustered)


# In[95]:


arr_clustered.shape


# **Problem** The image is not properly showing because of the wrong shape - 3,170,304 meaning one long line of pixels. Note that the y axis is up to 10e6. 
# **Solution** It should be 3-dimensional of (2048, 1548, 3)  - 2048 X 1548 representing the 2D picture.

# In[96]:


R, G, B = arr.shape
print(R, G, B)


# In[97]:


arr_clustered2 = arr_clustered.reshape([R, G, B])
print(type(arr_clustered2))
print("\nShape:", arr_clustered2.shape, '\n')
print(arr_clustered2[0])


# In[98]:


imshow(arr_clustered2)
plt.title('K=3 Clusters')


# ### Experiment with different clusters! 

# In[99]:


def one_stop_function(original_image, reshaped_image, k, starting_centers, max_steps):
    # Apply the k-means function
    cluster = kmeans(reshaped_image, k=k, starting_centers=None,max_steps=np.inf)
    assert len(set(cluster)) == k, "Number of identified clusters does not match the specified cluster numbers."
    
    # Calculate the mean of each cluster
    centers = {}
    for i in range(len(set(cluster))):
        centers[i] = np.mean(reshaped_image[cluster==i], axis = 0)
    
    # Generate a matrix 
    img_clustered = np.array([centers[i] for i in cluster])
    
    # Reshape the matrix to the shape of the original image
    R, G, B = original_image.shape
    return img_clustered.astype(dtype='uint8').reshape([R, G, B])


# In[100]:


k_2 = one_stop_function(original_image=img_arr,
                        reshaped_image=img_reshaped, k=2, starting_centers=None, max_steps=np.inf)

# Display the clustered image 
imshow(k_2)
plt.title('K=2 Clusters')


# In[101]:


k_5 = one_stop_function(original_image=img_arr,
                        reshaped_image=img_reshaped, k=5, starting_centers=None, max_steps=np.inf)

# Display the clustered image 
imshow(k_5)
plt.title('K=5 Clusters')


# In[103]:


k_10 = one_stop_function(original_image=img_arr,
                         reshaped_image=img_reshaped, k=10, starting_centers=None, max_steps=np.inf)

# Display the clustered image 
imshow(k_10)
plt.title('K=10 Clusters')


# ### Summary: 
# The processing time is not linear but exponential because of the big dataset.
# 
# - k=2 clusters took 14-16 iterations until convergence
# - k=3 clusters took 30-33 iterations until convergence
# - k=5 clusters took 70-85 iterations until convergence
# - k=10 clusters took 90-166 iterations until convergence
# 

# In[ ]:




