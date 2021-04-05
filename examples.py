from run_predictions import *

# Question 4: Sucessful examples
successes = [9, 10, 333]

for i in successes:
    I = Image.open(os.path.join(data_path,file_names[i]))
    im = np.asarray(I)

    bounding_boxes = detect_red_light(im)

    display_results(im, bounding_boxes)

# Question 5: Unsuccessful examples
failures = [27, 314, 183]

for i in failures:
    I = Image.open(os.path.join(data_path,file_names[i]))
    im = np.asarray(I)

    bounding_boxes = detect_red_light(im)

    display_results(im, bounding_boxes)