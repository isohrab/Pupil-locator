import os
import numpy as np
import cv2
from random import shuffle


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rf(low, high):
    """
    return a random float number between [low, high)
    :param low: lower bound
    :param high: higher bound (excluded)
    :return: a float number between [low, high)
    """
    if low >= high:
        return low
    return np.random.uniform(low, high)


def ri(low, high):
    """
    return a random int number between [low, high)
    :param low: lower bound
    :param high: higher bound (excluded)
    :return: an int number between [low, high)
    """
    if low >= high:
        return low
    return np.random.randint(low, high)


def anotator(img, label):
    """
    draw a yellow + on input image based label
    :param img: input image
    :param label: location of pupil
    :return: anotated pupil location
    """
    w, h = img.shape
    rgb = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    rgb[:, :, 0] = img  # TODO: a better way!
    rgb[:, :, 1] = img
    rgb[:, :, 2] = img

    # l1xs = int(label[0] - label[2] / 2)
    # l1ys = int(label[1])
    # l1xe = int(label[0] + label[2] / 2)
    # l1ye = int(label[1])
    #
    # l2xs = int(label[0])
    # l2ys = int(label[1] - label[3] / 2)
    # l2xe = int(label[0])
    # l2ye = int(label[1] + label[3] / 2)
    #
    # rgb = cv2.line(rgb, (l1xs, l1ys), (l1xe, l1ye), (255, 255, 0), 1)
    # rgb = cv2.line(rgb, (l2xs, l2ys), (l2xe, l2ye), (255, 255, 0), 1)

    # draw ellipse
    x = label[0]
    y = label[1]
    w = label[2]
    h = label[2]
    a = 0
    color = (0, 250, 250)
    rgb = cv2.ellipse(rgb, ((x, y), (w, h), a), color, 1)

    return rgb


def create_noisy_video(data_path='data/valid_data.csv', length=60, fps=5, with_label=False, augmentor=None):
    """
    create a sample video based random image.
    Of course it is not a valid solution to test the model with already seen images.
    It is just to check the speed of model. based on different FPS
    :param data_path: CSV file for input data
    :param length: length of video in second
    :param fps: number of frame per second
    :param with_label: if true, show true label on the video
    :return: a noisy video (file name) for test purpose.
    """

    # read CSV
    data_list = []
    with open(data_path, "r") as f:
        for line in f:
            #  values: [ img_path, x, y, w, h , a]
            values = line.strip().split(",")
            data_list.append([values[0],  # image path
                              values[1],  # x
                              values[2],  # y
                              values[3],  # w
                              values[4],  # h
                              values[5]])  # a

    # number image to make the video
    images_len = fps * length
    start_idx = np.random.randint(0, len(data_list) - images_len)
    selected_images = data_list[start_idx:start_idx + images_len]

    output_fn = 'video_{}s_{}fps.avi'.format(length, fps)
    # TODO: width/height are hard coded
    video = cv2.VideoWriter(output_fn, cv2.VideoWriter_fourcc(*"XVID"), fps, (192, 192))

    for i in selected_images:
        img = cv2.imread(i[0], cv2.IMREAD_GRAYSCALE)
        x = float(i[1])
        y = float(i[2])
        w = float(i[3])
        h = float(i[4])
        a = float(i[5])
        label = [x, y, w, h, a]
        if augmentor is not None:
            img, label = augmentor.addNoise(img, label)
            img = np.asarray(img, dtype=np.uint8)

        if with_label:
            img = anotator(img, label)
            font = cv2.FONT_HERSHEY_PLAIN
            texts = i[0].split("/")
            text = texts[1]+"/"+texts[2]+"/"+texts[3]
            img = cv2.putText(img, text, (5, 10), font, 0.8, (250, 0, 0), 1, cv2.LINE_8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        video.write(img)

    cv2.destroyAllWindows()
    video.release()

    return output_fn


def change_channel(frame, num_channel=1):
    """
    Get frame and normalize values between 0 and 1 and then based num channel reshape it to desired channel
    :param frame: the input image, a numpy array
    :param num_channel: desired number of channel
    :return: normalized frame with num_channel
    """
    img = frame / 255
    if num_channel == 3:
        w, h = img.shape
        img = np.tile(img, (3, 1))
        img = np.reshape(img, (w, h, 3))
    elif num_channel == 1:
        img = np.expand_dims(img, -1)
    else:
        raise ValueError("Are you sure?")

    return img

MIN_IMG_W = 192
def bound_it(amin, amax, imgmax):
    assert imgmax >= MIN_IMG_W
    s = amax - amin
    if s < MIN_IMG_W:
        d = MIN_IMG_W - s
        amin = amin - d / 2
        amax = amax + d / 2
        if amin < 0:
            amax += abs(amin)
            amin = 0

        if amax > imgmax:
            amin -= amax - imgmax
            amax = imgmax

            assert amin >= 0 and amax < imgmax
    return [amin, amax]

if __name__ == "__main__":
    print(bound_it(163,423,333))
