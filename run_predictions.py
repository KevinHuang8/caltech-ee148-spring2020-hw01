import os
import numpy as np
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def load_image(filename):
    img = Image.open(filename)
    img.load()
    return img

def normalize_image(image):
    '''
    z-score normalize, sample-wise
    '''
    image_norm = (image - image.mean()) / image.std()
    return image_norm

def convolve(image, kernel):
    '''
    image - n x m x 3 array (RGB)
    kernel - p x q x 3 array

    Returns the convolution of the image with the kernel.
    '''

    p, q, _ = kernel.shape
    n, m, _ = image.shape

    result = np.zeros((n - p + 1, m - q + 1))

    for i in range(n - p + 1):
        for j in range(m - q + 1):
            section = image[i:i+p, j:j+q, :1]
            dot_product = np.sum(section*kernel)
            result[i, j] = dot_product

    return result

def flood_fill(arr, limit=20, ecc_thresh=1.5):
    '''
    Takes in an array (corresponding to the result array after applying the
    kernel) that has values of either 0 or 255, with 255 corresponding to pixels
    that have been identified as part of a traffic light. This function clusters
    all of the pixels by assigning a different integer to each pixel group.

    limit - maximum number of pixels allowed in a cluster. If this is exceeded,
    that cluster is discarded, as traffic lights should not be very big 
    (they should not exceed the size of the filter).
    '''
    n = 1
    while 255 in arr:
        loc = np.argwhere(arr == 255)[0]
        flood_fill_helper(arr, loc, n, set())
        count = len(np.argwhere(arr == n))
        if count > limit:
            arr[arr == n] = 0
        if eccentricity(arr, n) > ecc_thresh:
            arr[arr == n] = 0
        n += 1
    return n - 1

def flood_fill_helper(arr, loc, cluster_num, visited):
    i, j = loc[0], loc[1]
    visited.add((i, j))
    if (arr[i, j] != 255):
        return
    arr[i, j] = cluster_num
    if (i + 1, j) not in visited and i < arr.shape[0] - 1:
        flood_fill_helper(arr, (i + 1, j), cluster_num, visited)
    if (i, j + 1) not in visited and j < arr.shape[1] - 1:
        flood_fill_helper(arr, (i, j + 1), cluster_num, visited)
    if (i - 1, j) not in visited and i > 0:
        flood_fill_helper(arr, (i - 1, j), cluster_num, visited)
    if (i, j - 1) not in visited and j > 0:
        flood_fill_helper(arr, (i, j - 1), cluster_num, visited)

def eccentricity(arr, n):
    '''
    Returns the "eccentricity" of cluster n in arr. Eccentricity is defined
    as the ratio of the range of the cluster along axis to the range along the
    other axis. Traffic light clusters are roughly circular, so high
    eccentricities should be discarded.
    '''
    matches = np.transpose((arr == n).nonzero())
    if matches.size == 0:
        return 1

    mins = np.min(matches, axis=0)
    maxes = np.max(matches, axis=0)
    
    dx = np.abs(mins[1] - maxes[1]) + 1
    dy = np.abs(mins[0] - maxes[0]) + 1

    larger = max(dx, dy)
    smaller = min(dx, dy)

    return larger / smaller

def stop_overlap(bounding_boxes):
    '''
    Takes a list of bounding boxes and removes any overlapping ones.
    '''
    invalid = set()
    for i, (xmin1, ymin1, xmax1, ymax1) in enumerate(bounding_boxes):
        if i in invalid:
            continue
        for j, (xmin2, ymin2, xmax2, ymax2) in enumerate(bounding_boxes):
            if i >= j:
                continue

            if j in invalid:
                continue

            if overlap_1D((xmin1, xmax1), (xmin2, xmax2)) and \
                overlap_1D((ymin1, ymax1), (ymin2, ymin2)):
                invalid.add(j)
                invalid.add(i)
                continue

    new_boxes = []
    for i, box in enumerate(bounding_boxes):
        if i not in invalid:
            new_boxes.append(box)
    return new_boxes

def overlap_1D(interval1, interval2):
    return (interval1[1] >= interval2[0] and interval2[1] >= interval1[0]) \
        or (interval2[1] >= interval1[0] and interval1[1] >= interval2[0])

def get_centers(arr, n_clusters):
    '''
    Returns a list of coordinates corresponding to the centers for each
    out of n_clusters clusters in arr.
    '''
    centers = []
    for i in range(1, n_clusters + 1):
        matches = np.transpose((arr == i).nonzero())
        if matches.size == 0:
            continue
        centers.append(np.mean(matches, axis=0))
    return centers

def _detect_red_light(image, kernel, quantile, display=False):
    '''
    Helper function.
    '''

    bounding_boxes = []
    
    image_norm = normalize_image(image)
    result = convolve(image_norm, kernel)

    threshold = np.quantile(result, quantile)

    result[result < threshold] = 0
    result[result >= threshold] = 255

    n_clusters = flood_fill(result, limit=kernel.shape[0]*kernel.shape[1]*0.5)
    centers = get_centers(result, n_clusters)

    for i, j in centers:
        bounding_boxes.append([i, j, i + kernel.shape[0], j + kernel.shape[1]])
    bounding_boxes = stop_overlap(bounding_boxes)

    # Debug
    if display:
        im = Image.fromarray(image.astype('uint8'), 'RGB')
        draw = ImageDraw.Draw(im)
        for i0, j0, i1, j1 in bounding_boxes:
            draw.rectangle((j0, i0, j1, i1), outline='red')
        im.show()

        plt.imshow(result)
        plt.show()

    return bounding_boxes

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    # Paramters for the algorithm
    scaling_factors = [1, 1/2, 1/3]
    quantile = 0.9995
    attempts_per_scale = 2
    kernel = load_image('filter.png')

    bounding_boxes = []
    kernel = np.asarray(kernel, dtype='int32')[:,:,:3]
    kernel_shape = kernel.shape

    # Try different scale kernels, stopping if enough traffic lights
    # have been found
    for s in scaling_factors:
        kernel = load_image('filter.png')
        if s != 1:
            kernel = kernel.resize((int(kernel_shape[1]*s), int(kernel_shape[0]*s)))
        kernel = np.asarray(kernel, dtype='int32')[:,:,:3]
        kernel = normalize_image(kernel)

        bb = _detect_red_light(I, kernel, quantile=quantile, display=False)
        if len(bb) >= attempts_per_scale:
            bounding_boxes.extend(bb)
            break

        bounding_boxes.extend(bb)
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    # im = Image.fromarray(I.astype('uint8'), 'RGB')
    # draw = ImageDraw.Draw(im)
    # for i0, j0, i1, j1 in bounding_boxes:
    #     draw.rectangle((j0, i0, j1, i1), outline='red')
    # im.show()
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = 'data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

if __name__ == '__main__':
    preds = {}
    for i in range(len(file_names)):
        print(i)
        i = 9
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names[i]))
        
        # convert to numpy array:
        I = np.asarray(I)
        
        preds[file_names[i]] = detect_red_light(I)
        break
        
    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds.json'),'w') as f:
        json.dump(preds,f)
