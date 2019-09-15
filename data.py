import h5py
import numpy as np


def get_data(dim_x, dim_y, year=2005):
    data_path = "./climo_" + str(year) + ".h5"
    h5f = h5py.File(data_path)
    images = h5f["images"]
    boxes = h5f["boxes"]

    # Pre-processing
    x = get_x(images, dim_x, dim_y)
    y = get_y(boxes, dim_x[0], dim_x[0], dim_y[1], dim_y[1], year=year)

    return x, y


def get_x(images, dim_x, dim_y):
    x = images[:, :, dim_y[0]:dim_y[1], dim_x[0]:dim_x[1]]
    x = np.moveaxis(x, 1, -1)
    mean = np.mean(x.reshape((-1, 16)), axis=0).reshape(1, 1, 1, 16)
    std = np.std(x.reshape((-1, 16)), axis=0).reshape(1, 1, 1, 16)
    x = (x - mean) / std
    return x


def get_y(boxes_image, min_x, min_y, max_x, max_y, year=2005):
    def return_y_label(label_index, boxes_image, min_y, min_x, max_y, max_x):
        list_number_event = []
        if year == 2005:
            range_val = 1456
        if year == 2004:
            range_val = 1460
        for i in range(range_val):
            boxes_image_2 = boxes_image[i, :, :]
            cal = boxes_image_2[boxes_image_2[:, 4] == label_index]
            cal_or = boxes_image_2[boxes_image_2[:, 4] == label_index]
            area_original = (cal[:, 2] - cal[:, 0]) * (cal[:, 3] - cal[:, 1])
            cal[cal[:, 0] < min_y] = min_y
            cal[cal[:, 2] < min_y] = min_y
            cal[cal[:, 0] > max_y] = max_y
            cal[cal[:, 2] > max_y] = max_y
            cal[cal[:, 1] < min_x] = min_x
            cal[cal[:, 3] < min_x] = min_x
            cal[cal[:, 1] > max_x] = max_x
            cal[cal[:, 3] > max_x] = max_x
            area_new = (cal[:, 2] - cal[:, 0]) * (cal[:, 3] - cal[:, 1])
            fraction = area_new / area_original
            cal = cal[fraction > 0.5]
            # cal = cal[(cal[:,0]>min_y) & (cal[:,1]>min_x) & (cal[:,2]<max_y) & (cal[:,3]< max_x)]
            list_number_event += [cal.shape[0]]
        y_label = [1 if i != 0 else 0 for i in list_number_event]
        return y_label

    all_y = []
    for label in range(4):
        all_y += [return_y_label(label, boxes_image, min_y, min_x, max_y, max_x)]
    all_y = np.array(all_y).T
    return all_y
