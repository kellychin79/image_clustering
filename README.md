## Introduction

K-means clustering is a very popular machine-learning classification algorithm. It is an **unsupervised** algorithm because you don’t need to know the answers (labels) beforehand. 

It starts with calculating the distance between all data points and arbitrary starting points. After iterations, it finds out the optimal clusters that divide the distinctive data points into more general groups. 

## Motivation

I experimented on numerous projects using K-means, such as the classic iris dataset, gambler dataset, and homeowner dataset. So far, image clustering is the most interesting one - it is a primitive form of image filter!

In this project, I wanted to transform a photo of me, holding a US flag in a red dress on the Fourth of July, into a photo of 2 colors (black and white), 3 colors, 5 colors, and 10 colors. Let's get started!


A photo has millions of pixels, each of which represents a unique **RGB data value** (Red, Green, Blue). Below, you can find different ideal image sizes in terms of resolution.

- Source: https://www.adobe.com/uk/creativecloud/photography/discover/standard-photo-sizes.html#:~:text=An%20image%20size%20of%201280,is%20also%20common%20in%20filmmaking
	- 4 x 6” 1200 x 1800 pixels = 2.2 million pixels
	- 5 x 7” 1500 x 2100 pixels = 3.2 million pixels
	- 8 x 10 “ 2400 x 3000 pixels = 7.2 million pixels
	- Instagram posts 1080 x 1080 pixels = 1.2 million pixels 
	- Pinterest pins 1000 x 1500 pixels = 1.5 million pixels

- Source: https://www.winxdvd.com/streaming-video/tiktok-video-resolution-dimensions.htm
	- TikTok videos 1080×1920 pixels = 2.1 million pixels



## Descriptions

First of all, I reshaped the original 3-dimensional image file (2,048 x 1,548 x 3) into a very long 2-dimensional array (3,170,304 x 3). Note the 3 stores the RGB values respectively.

Next, I applied the K-means clustering algorithm. It went through the array of RGB data values, calculated for iterations until convergence, and lastly classified similar ones into the same group. 


## Summary 
The processing time is not linear but exponential because of the big dataset. Here are the observations: k=2 clusters took 14-16 iterations until convergence (black and white); k=3 clusters took 30-33 iterations until convergence; k=5 took 70-85 iterations; k=10 took 90-166 iterations (almost carrying the original style). 

Thousands of colors in the photo were grouped and compressed - dark, bright, dim, and vibrant. Accordingly, the image style amazingly changed - just like the filters in Snapseed, Instagram, and other photo-editing apps!

<img src="pic.jpeg" width="50%" height="50%" />

![alt text](https://github.com/kellychin79/image_clustering/blob/main/output.png?raw=true) 


