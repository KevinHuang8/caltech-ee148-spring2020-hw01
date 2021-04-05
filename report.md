Github link: https://github.com/KevinHuang8/caltech-ee148-spring2020-hw01

**Deliverable #1**

1. The primary algorithm was a matched filtering algorithm as described in lecture 2, with some additional steps. The algorithm can be described as follows:

   1. Let the kernel $k$ be an arbitrary image of a red light. 
   2. For each image, perform z-score normalization sample-wise. 
   3. Convolve the image $I$ with $k$, to produce an output $r$.
   4. Take the $\alpha$, $\alpha \in (0, 1)$, percentile of pixels in $r$ to be the candidate locations for a red light. Set all other pixels to 0.
   5. Cluster the nonzero pixels in $r$ by labeling all contiguous nonzero pixels by unique identifier. This is done by a simple flood fill algorithm.
   6. Compute the centers of each cluster, which are chosen as the locations of each traffic light. The bounding boxes are the size of the kernel. 

   All attempts were modifications/additions to this basic algorithm. First, different values of the threshold $\alpha$ were tested.

   The second modification was to account for the different scales of traffic lights that appear in different images. A list of scales $[s_1, s_2, \dots, s_n]$ was defined as a parameter for our algorithm, with $1 >= s_i > s_{i + 1}$. The main algorithm was then repeated with the kernel being rescaled by factor of $s_i$, for $i = 1, \dots, n$. If a kernel scale produced a result with more than some threshold $t$ traffic lights, this process stops, as it probably has reached the correct scale.

   The third modification was to limit the size of clusters in step 5 above. That is, we know that traffic lights should not be bigger than the kernel itself (which is an entire traffic light). Thus, if we get a region of $r$ that is larger than the kernel, then it is likely a false positive, and so we ignore it.

   The fourth modification was to limit the *eccentricity* of clusters in step 5. This comes from the observation that red lights should produce a circular pattern when convolved with the kernel (as the light is circular). Thus, if the shape of our clusters is very different from circular, we discard that location as a false positive.

   The final modification was to eliminate bounding boxes that overlap, as traffic lights likely do not overlap. This may introduce some false negatives, however.

2. Algorithm performance was evaluated empirically by looking at the rates of false positives and false negatives on a variety of example images. I wanted to balance the two so that there weren't too many extraneous bounding boxes, but also so that the algorithm correctly detected at least some of the red lights in each image. With the relatively basic sophistication of the algorithms, I was willing to accept if objects similar to red traffic lights were detected, such as red lights from cars. Thus, the algorithm often incorrectly marks car lights as traffic lights, which is misleading, but I gave those a pass since they were similar enough to red traffic lights. 

3. The best algorithm included all of the modifications included above, with $\alpha = 0.9995$. The scales used were $[1, \frac12, \frac13]$, with $t = 2$. That is, the best algorithm was a matched filtering algorithm with additional heuristics to remove false positives. 

4. 

   Successful examples:

   <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210404220039677.png" alt="image-20210404220039677" style="zoom:50%;" />![image-20210404220100115](C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210404220100115.png)

<img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210404220100115.png" style="zoom:50%;" />

<img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210404220119150.png" alt="image-20210404220119150" style="zoom:50%;" />

These examples were easy because the red lights were very prominent in the image, and also closely matched the size of the kernel. All of the red lights were also oriented head on, so all the red lights were detected. Furthermore, there were few other "distractions" in the images; that is, the images were not too busy with other things that could potentially look like red lights and confuse the algorithm, and so there were no false positives. However, we see that the bounding boxes are not perfect, as the kernel does not precisely match the shape of the traffic lights. 

5.

Unsuccessful examples:

<img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210404220508026.png" alt="image-20210404220508026" style="zoom:50%;" />![image-20210404220524788](C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210404220524788.png)



<img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210404220629883.png" alt="image-20210404220629883" style="zoom:50%;" />

<img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210404220650023.png" alt="image-20210404220650023" style="zoom:50%;" />



The red light kernel seems to match with anything that is light on the top third, and dark on the bottom third, making it very prone to false negatives. For example, from the above examples, we see that a dark tree with light blue sky in the background can often get confused with a red light, since the top half is light and the bottom is dark. And because the algorithm first uses a big scale and only checks for smaller traffic lights if few big ones are found, a couple of false positives on a larger scale can lead to smaller lights not being found, which we see with the first example. This is a flaw in the algorithm.

Furthermore, the presence of many different sources of red lights can confuse the algorithm and result in false negatives, for example car lights, as we see in the third example. The red lights' reflections on the car hood also are identified as traffic lights, as the algorithm cannot tell the difference between reflections and real objects. 

6. As mentioned earlier, a big problem is with the scale of the traffic lights, since the traffic lights appear in multiple different sizes in different images, while the matched filtering algorithm can only use one fixed size kernel. The current approach performs multiple convolutions with different size kernels, working its way down in size. One problem in the implementation is that the algorithm arbitrarily stops once it has found enough traffic lights, but if there are false positives with the large kernels, then it may never find smaller red traffic lights. 

​		One solution that still uses matched filtering would be to instead try all size kernels, but then for each point in the result $r$, to take the maximum of the dot product for each size kernel, weighted by the size of the kernel. This would make it so that hopefully the best size kernel is chosen for each point. The problem still remains that the kernel sizes would have to be manually selected, but it should still be an improvement over the current algorithm. 