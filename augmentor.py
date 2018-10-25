import os

import cv2
import numpy as np

from config import config
from utils import rf, ri, create_noisy_video


def assert_it(img, lbl):
    """
    Add assertation between augmentation operation
    :param img:
    :param lbl:
    :return:
    """
    # Check input image
    assert_msg = "input image should be HxW, with type of np.uint8"
    assert isinstance(img, np.ndarray), assert_msg
    assert np.ndim(img) == 2, assert_msg
    assert img.dtype == np.uint8, assert_msg

    # get the input image shape
    h, w = img.shape
    assert h == w, "Input image must have same width and height"

    # check input label
    assert isinstance(lbl, list), "Label should be a list"
    assert len(lbl) == 5, "Length of label should be 5 (x, y, w, h, a)"
    assert (0 <= lbl[0] <= w), "x value should be in range of 0 and width of image"
    assert (0 <= lbl[1] <= h), "y value should be in range of 0 and height of image"

    return


class Augmentor(object):
    """
    add noise to the images
    """

    def __init__(self, noise_dir, noise_parameters):
        self.noise_dir = noise_dir
        self.cfg = noise_parameters

        # check if the noisy videos are exist
        if not os.path.isdir(noise_dir):
            raise FileNotFoundError

        # read all videos
        videos_fn = [os.path.join(self.noise_dir, f)
                     for f in os.listdir(self.noise_dir)
                     if f.endswith(".mp4")]

        # read all frames and load them into memory
        self.frames = []
        for video in videos_fn:
            print("loading video {}".format(video))
            cap = cv2.VideoCapture(video)
            ret = True
            while ret:
                ret, frame = cap.read()
                if ret:
                    frame = frame[100:, 50:]
                    frame = cv2.resize(frame, (2 * self.cfg["input_height"], 2 * self.cfg["input_width"]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.frames.append(frame)

            cap.release()

        print("In total {} frames loaded".format(len(self.frames)))

    def downscale(self, img, label):
        """
        Downscale the input image to a random value defined in the config file
        :param img: input image
        :param label: input label
        :return: return downscaled image and updated ground truth
        """
        # should we upscale the input image?
        if self.cfg["prob_downscale"] < rf(0, 1):
            return img, label

        # get a random scale value
        s = rf(self.cfg["min_downscale"], self.cfg["max_downscale"])
        out_img = cv2.resize(img, dsize=(0, 0), fx=s, fy=s)

        # get a random frame as background
        idx = ri(0, len(self.frames))
        bg = self.frames[idx]
        bg = cv2.resize(bg, dsize=(config["input_height"], config["input_width"]))

        # put scaled image somewhere in the background
        h, w = img.shape
        s_h, s_w = out_img.shape

        dw = w - s_w
        dh = h - s_h

        # random location
        rx = ri(0, dw)
        ry = ri(0, dh)

        # put it on the background frame
        bg[ry:ry + s_h, rx:rx + s_w] = out_img

        # update the label based movement and scale
        lx = label[0] * s + rx
        ly = label[1] * s + ry
        lw = label[2] * s

        # clip the values inside the image bound (height, widht)
        lx = np.clip(lx, 0, w)
        ly = np.clip(ly, 0, h)

        return bg, [lx, ly, lw]

    def addReflection(self, in_img):
        """
        Reflect a random noisy frame on the image
        :param in_img: input image
        :return: image + reflection
        """
        # should we add reflection to the input?
        if self.cfg["prob_reflection"] < rf(0, 1):
            return in_img

        # randomly select a reflection from frames
        idx = ri(0, len(self.frames))
        frame = self.frames[idx]

        # the size of noisy frame is bigger than input image. we choose a random location to crop the noisy
        # frame with the size equal to input image
        sx = ri(0, config["input_width"])
        sy = ri(0, config["input_height"])
        ref = frame[sy:sy + config["input_height"], sx:sx + config["input_width"]]

        # choose a random weight: read the paper for the details
        max_beta = rf(self.cfg["min_reflection"], self.cfg["max_reflection"])
        beta = ref / 255
        neg = (in_img / 255) - 0.75
        beta = beta + neg
        beta = np.clip(beta, 0, max_beta)
        res = in_img + beta * (255.0 - in_img) * (ref / 255.0)
        return np.asarray(res, dtype=np.uint8)

    def addBlur(self, in_img):
        """
        add gaussian blur to the input image
        :param in_img: input image
        :return: blured image
        """
        if self.cfg["prob_blur"] < rf(0, 1):
            return in_img

        ksize = ri(self.cfg["min_blurSize"], self.cfg["max_blurSize"])
        if ksize % 2 == 0:
            ksize = ksize + 1
        sigma = rf(self.cfg["min_sigmaRatio"], self.cfg["max_sigmaRatio"])
        return cv2.GaussianBlur(in_img, (ksize, ksize), sigma)

    def addOcclusion(self, in_img, in_label):
        """
        erase some part of pupil area
        :param in_img: input image
        :param in_label: just use pupil location
        :return: erased image
        """
        if self.cfg["prob_occlusion"] < rf(0, 1):
            return in_img

        # randomly choose # object on the eye
        num_obj = ri(0, self.cfg["occlusion_max_obj"])

        # shorthand the w h
        p_x = int(in_label[0])
        p_y = int(in_label[1])
        p_w = int(in_label[2] * 1.5)
        p_h = int(in_label[3] * 1.5)

        # choose a random size of the object
        obj_w = int(p_w * rf(self.cfg["min_occlusion"], self.cfg["max_occlusion"]))
        obj_h = int(p_h * rf(self.cfg["min_occlusion"], self.cfg["max_occlusion"]))

        # choose a random location around the pupil
        x_area = np.clip(p_x - p_w + ri(0, p_w), 0, self.cfg["input_width"])
        y_area = np.clip(p_y - p_h + ri(0, p_h), 0, self.cfg["input_height"])

        # choose a random color based the current pupil color
        occ_color = ri(245, 256)

        # add object in random place close together
        for i in range(num_obj):
            obj_x = np.clip(x_area + ri(0, obj_w * 2), 0, self.cfg["input_width"] - obj_w)
            obj_y = np.clip(y_area + ri(0, obj_h * 2), 0, self.cfg["input_height"] - obj_h)

            # create a occlusion matrix
            o = np.ones((obj_h, obj_w), dtype=np.uint8) * occ_color

            # put occlusion inside the img
            in_img[obj_y:obj_y + obj_h, obj_x:obj_x + obj_w] = o

        return in_img

    def addPupil(self, _img, _lbl, max_attemps=100):
        """
        Add a pupil-like ellipse on the image.
        :param _img: input image
        :param _lbl: use current ground truth info for new pupil
        :return:
        """
        if self.cfg["prob_pupil"] < rf(0, 1):
            return _img

        # read the ground-truth info
        x = _lbl[0]
        y = _lbl[1]
        w = _lbl[2]

        attemps = 0

        # try this # max_attemos
        while attemps < max_attemps:
            attemps += 1
            # choose randomly new location
            lx = ri(0, self.cfg["input_width"])
            ly = ri(0, self.cfg["input_height"])
            lw = ri(w / 2, w * 1.2)
            lh = ri(w / 2, w * 1.5)
            la = ri(0, 180)

            # calculate the distance between real pupil and new one, not overlapping
            d = np.sqrt((x - lx) ** 2 + (y - ly) ** 2)
            if d < w:
                continue

            # get the color of new pupil based on current pupil
            c = int(_img[int(y), int(x)])
            c = ri(c * 0.7, c * 1.2)
            # draw an ellipse on the image
            img = cv2.ellipse(_img, ((lx, ly), (lw, lh), la), (c), -1)

            return img

        # if we are here, max_attmeps reached
        return _img

    def addExposure(self, in_img):
        """
        Add exposure to image
        :param in_img: input image
        :return: exposured image
        """
        if self.cfg["prob_exposure"] < rf(0, 1):
            return in_img

        # get a random exposure value based on max-min value in config file
        exp_val = rf(self.cfg["min_exposure"], self.cfg["max_exposure"])
        in_img = in_img * exp_val
        in_img = np.clip(in_img, 0, 255)
        in_img = np.asarray(in_img, dtype=np.uint8)
        return in_img

    def crop_it(self, img, lbl, max_attemps=100):
        """
         crop the input image with a random location and size.
        :param img: input size
        :param label: location of pupil
        :return: cropped image + new label based on crop
        """
        if config["crop_probability"] < rf(0, 1):
            return img, lbl

        # get the shape of image
        h, w = img.shape

        # get the labels
        lx = lbl[0]
        ly = lbl[1]
        lw = lbl[2]

        # find pupil upper right corner and bottom left corner to check if
        # it is in the cropped image or not, we consider pupil is circle and use only width
        px1 = lx - lw / 2
        py1 = ly - lw / 2
        px2 = lx + lw / 2
        py2 = ly + lw / 2
        # check if pupil location is not outside of the image
        px1, py1, px2, py2 = np.clip([px1, py1, px2, py2], 0, w)

        attemps = 0
        while attemps < max_attemps:
            # create a random size
            crop_size = int(rf(config["crop_min_ratio"], config["crop_max_ratio"]) * w)

            # choose a point in top left corner
            cx1 = ri(0, w - crop_size)
            cy1 = ri(0, w - crop_size)

            # bottom right corner
            cx2 = cx1 + crop_size
            cy2 = cy1 + crop_size

            # check if pupil is out side of crop
            if px1 < cx1 or px1 > cx2:
                attemps += 1
                continue

            if px2 < cx1 or px2 > cx2:
                attemps += 1
                continue

            if py1 < cy1 or py1 > cy2:
                attemps += 1
                continue

            if py2 < cy1 or py2 > cy2:
                attemps += 1
                continue

            # if we are here, it means we found a crop box
            # slice the image
            image = img[cy1:cy1 + crop_size, cx1:cx1 + crop_size]

            # update the label for crop
            lx = lx - cx1
            ly = ly - cy1

            # resize back to input size
            image = cv2.resize(image, dsize=(config["input_height"], config["input_width"]))

            # update the labels
            s = config["input_width"] / crop_size
            lx = lx * s
            ly = ly * s
            lw = lw * s

            return image, [lx, ly, lw]

        # if we are here, no crop applied
        return img, lbl

    def flip_it(self, img, lbl):
        """
        flip an image right to left
        :rtype: (np.ndarray, list)
        :param img: input image
        :param lbl: input label
        :return: flipped image + altered label
        """
        if config["flip_probability"] < rf(0, 1):
            return img, lbl

        h, w = img.shape
        img = cv2.flip(img, 1)

        # update the label
        lx = w - lbl[0]
        ly = lbl[1]
        lw = lbl[2]

        return img, [lx, ly, lw]

    def resize_it(self, img, lbl):
        """
        get an image with different size and convert it to Model input size.
        Model input size defined in config file. Also update corresponding label
        :param img:
        :param lbl:
        :return: resized image and updated label
        """

        h, w = img.shape

        # calculate the scale factor.
        s = config["input_width"] / w

        # Resize the input image, w and h must be same
        new_img = cv2.resize(img, dsize=(config["input_width"], config["input_width"]))

        # update the labels based new size
        lx = lbl[0] * s
        ly = lbl[1] * s
        lw = lbl[2] * s
        lh = lbl[3] * s
        la = lbl[4]

        return new_img, [lx, ly, lw, lh, la]

    def addNoise(self, in_img, in_label):
        """
        Add all possible noise to the image
        :param in_img: input image
        :param in_label: pupil location
        :return: return augmented image
        """
        # first make a copy of image and labels
        c_img = np.array(in_img, copy=True)
        c_label = list(np.array(in_label, copy=True))

        # apply noise
        c_img = self.addPupil(c_img, c_label)

        c_img = self.addExposure(c_img)

        c_img, c_label = self.flip_it(c_img, c_label)
        assert_it(c_img, c_label)
        #
        c_img, c_label = self.downscale(c_img, c_label)
        assert_it(c_img, c_label)
        #
        c_img, c_label = self.crop_it(c_img, c_label)
        assert_it(c_img, c_label)
        #
        c_img = self.addReflection(c_img)
        assert_it(c_img, c_label)

        c_img = self.addBlur(c_img)

        return c_img, c_label


if __name__ == "__main__":

    ag = Augmentor('data/noisy_videos/', config)
    create_noisy_video(length=50, fps=1, with_label=True, augmentor=ag)
