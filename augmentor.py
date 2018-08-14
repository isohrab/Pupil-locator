import os
import numpy as np
from config import config
from xml.etree import ElementTree
from utils import rf, ri, create_noisy_video
import cv2


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
    assert (0 < lbl[0] < w), "x value should be in range of 0 and width of image"
    assert (0 < lbl[1] < h), "y value should be in range of 0 and height of image"

    return


class Augmentor(object):
    """
    add noise to the data
    """

    def __init__(self, noise_dir, noise_parameters):
        self.noise_dir = noise_dir
        self.cfg = noise_parameters

        # extract the noise video from folder
        if not os.path.isdir(noise_dir):
            raise FileNotFoundError

        videos_fn = [os.path.join(self.noise_dir, f)
                     for f in os.listdir(self.noise_dir)
                     if f.endswith(".mp4")]

        # load videos to memory
        self.frames = []
        for video in videos_fn:
            print("loading video {}".format(video))
            cap = cv2.VideoCapture(video)
            ret, frame = cap.read()
            frame = frame[100:, 50:]
            frame = cv2.resize(frame, (self.cfg["input_height"], self.cfg["input_width"]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames.append(frame)
            while ret:
                ret, frame = cap.read()
                if ret:
                    frame = frame[100:, 50:]
                    frame = cv2.resize(frame, (self.cfg["input_height"], self.cfg["input_width"]))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.frames.append(frame)

            cap.release(); break

        print("In total {} frames loaded".format(len(self.frames)))

    def upscale(self, img, label):
        """
        upscale the input image. the default input image size is 192x192.
        we upscale it with random scale factor to maximum of 2.2. Consequencly upscale
        the label too.
        :param img: the input image
        :param label: the label of image (x,y,w,h,a)
        :return: scaled image with filled black border
        """
        # should we upscale the input image?
        if self.cfg["prob_upscale"] < rf(0, 1):
            return img, label

        # get a random scale value
        s = rf(self.cfg["max_upscale"], self.cfg["min_upscale"])
        out_img = cv2.resize(img, dsize=(0, 0), fx=s, fy=s)

        h, w = out_img.shape

        # update the label based movement and scale
        new_label = []
        x = label[0] * s
        y = label[1] * s
        w = label[2] * s
        h = label[3] * s
        a = label[4]

        return out_img, [x, y, w, h, a]

    def addReflection(self, in_img):
        """
        add a random reflection to the image
        :param in_img: input image
        :return: image + reflection
        """
        # should we add reflection to the input?
        if self.cfg["prob_reflection"] < rf(0, 1):
            return in_img

        # randomly select a reflection from frames
        idx = ri(0, len(self.frames))
        ref = self.frames[idx]

        # add exposure to the frame
        ref = self.addExposure(ref)

        h, w = in_img.shape
        rh, rw = ref.shape

        # get difference
        dw = rw - w
        dh = rh - h

        # get a random point between 0 and difference in each axis
        px = ri(0, dw)
        py = ri(0, dh)

        # slice the reflection frame into new image
        new_ref = ref[py:py + h, px:px + w]

        # choose a random weight
        beta = rf(self.cfg["min_reflection"], self.cfg["max_reflection"])
        res = in_img + beta * (255.0 - in_img) * (new_ref / 255.0)
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
        # if self.cfg["prob_occlusion"] < rf(0, 1):
        #     return in_img

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

        occ_color = ri(245, 256)
        for i in range(num_obj):
            obj_x = np.clip(x_area + ri(0, obj_w * 2), 0, self.cfg["input_width"] - obj_w)
            obj_y = np.clip(y_area + ri(0, obj_h * 2), 0, self.cfg["input_height"] - obj_h)

            # create a occlusion matrix
            o = np.ones((obj_h, obj_w), dtype=np.uint8) * occ_color

            # put occlusion inside the img
            in_img[obj_y:obj_y + obj_h, obj_x:obj_x + obj_w] = o

        return in_img

    def addExposure(self, in_img):
        """
        Add exposure to image
        :param in_img: input image
        :return: exposured image
        """
        if self.cfg["prob_exposure"] < rf(0, 1):
            return in_img

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
        lh = lbl[3]
        la = lbl[4]

        # find pupil upper right corner and bottom left corner to check if
        # it is in the cropped image or not, we consider pupil is circle and use only width
        px1 = lx - lw / 2
        py1 = ly - lh / 2
        px2 = lx + lw / 2
        py2 = ly + lh / 2
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

            return image, [lx, ly, lw, lh, la]

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
        lh = lbl[3]
        la = lbl[4] * -1

        return img, [lx, ly, lw, lh, la]

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
        c_img, c_label = self.flip_it(c_img, c_label)
        assert_it(c_img, c_label)

        c_img, c_label = self.upscale(c_img, c_label)
        assert_it(c_img, c_label)

        c_img, c_label = self.crop_it(c_img, c_label)
        assert_it(c_img, c_label)

        c_img = self.addReflection(c_img)
        assert_it(c_img, c_label)

        c_img, c_label = self.resize_it(c_img, c_label)
        assert_it(c_img, c_label)

        return c_img, c_label


if __name__ == "__main__":
    # image_fn = "0in.jpg"
    # img = cv2.imread(image_fn, 0)
    # xml_path = "0gt.xml"
    # e = ElementTree.parse(xml_path).getroot()
    # x = np.round(np.float32(e[0].text))
    # y = np.round(np.float32(e[1].text))
    # w = np.round(np.float32(e[2].text))
    # h = np.round(np.float32(e[3].text))
    # a = np.round(np.float32(e[4].text))
    # true_label = [x, y, w, h]

    ag = Augmentor('data/noisy_videos/', config)
    create_noisy_video(length=200, fps=1, with_label=True, augmentor=ag)

    # pil_img = Image.fromarray(annotator(scaled_img, *scaled_label))
    # pil_img.show()
    #
    # pil_img = Image.fromarray(annotator(img, *true_label))
    # pil_img.show()
    # print("true label {}".format(true_label))
    # print("scaled label {}".format(scaled_label))

    # ag = Augmentor('data/noisy_videos/', config)
    #
    # img = np.random.randint(0,255, size=(20,20), dtype=np.uint8)
    # label = [8, 14, 3]
    # img, lbl = ag.crop_it(img, label)
