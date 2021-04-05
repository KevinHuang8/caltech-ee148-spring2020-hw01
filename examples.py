from run_predictions import *

# Question 4: Sucessful examples
# successes = [9, 10, 333]

# for i in successes:
#     I = Image.open(os.path.join(data_path,file_names[i]))
#     draw = ImageDraw.Draw(I)
#     im = np.asarray(I)

#     bounding_boxes = detect_red_light(im)

#     for i0, j0, i1, j1 in bounding_boxes:
#         draw.rectangle((j0, i0, j1, i1), outline='red')
#     I.show()

# Question 5: Unsuccessful examples
failures = [27, 314, 183]

for i in failures:
    I = Image.open(os.path.join(data_path,file_names[i]))
    draw = ImageDraw.Draw(I)
    im = np.asarray(I)

    bounding_boxes = detect_red_light(im)

    for i0, j0, i1, j1 in bounding_boxes:
        draw.rectangle((j0, i0, j1, i1), outline='red')
    I.show()