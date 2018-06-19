from utils import rf, ri, anotator
import os
import numpy as np
from PIL import Image
from config import config
from xml.etree import ElementTree
import cv2


class Augmentor(object):
    """
    add noise to the data
    """

    # TODO: need to crop the black spot from source video
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
            frame = cv2.resize(frame, (192, 192))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frames.append(frame)
            while ret:
                ret, frame = cap.read()
                if ret:
                    frame = frame[100:, 50:]
                    frame = cv2.resize(frame, (192, 192))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.frames.append(frame)

            cap.release()

        print("In total {} frames loaded".format(len(self.frames)))

    def downscale(self, img, label):
        """
        downscale input image
        :param img: the input image
        :param label: the label of image (x,y,w,h)
        :return: scaled image with filled black border
        """
        # should we downscale the input image?
        if self.cfg["prob_downscale"] < rf(0, 1):
            return img, label

        # create a random matrix
        # z = np.random.randint(0, 255, size=img.shape, dtype=np.uint8)

        idx = ri(0, len(self.frames))
        z = self.frames[idx]

        # get a random scale value
        s = rf(self.cfg["max_downscale"], self.cfg["min_downscale"])
        out_img = cv2.resize(img, dsize=(0, 0), fx=s, fy=s)

        # choose a location to put the scaled image inside the z matrix
        size_dif = z.shape[0] - out_img.shape[0]

        rOffset = ri(0, size_dif)
        cOffset = ri(0, size_dif)

        rEnd = rOffset + out_img.shape[0]
        cEnd = cOffset + out_img.shape[1]

        z[rOffset:rEnd, cOffset:cEnd] = out_img

        # update the label based movement and scale
        update_label = label[:]
        update_label[0] = label[0] * s + cOffset
        update_label[1] = label[1] * s + rOffset
        update_label[2] = label[2] * s
        update_label[3] = label[3] * s

        return z, update_label

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

        # choose a random weight
        w = rf(self.cfg["min_reflection"], self.cfg["max_reflection"])
        res = in_img + w * (255.0 - in_img) * (ref / 255.0)
        return res

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
        x_area = np.clip(p_x - p_w + ri(0, p_w), 0, self.cfg["image_width"])
        y_area = np.clip(p_y - p_h + ri(0, p_h), 0, self.cfg["image_height"])

        occ_color = ri(245, 256)
        for i in range(num_obj):
            obj_x = np.clip(x_area + ri(0, obj_w * 2), 0, self.cfg["image_width"]-obj_w)
            obj_y = np.clip(y_area + ri(0, obj_h * 2), 0, self.cfg["image_height"]-obj_h)

            # create a occlusion matrix
            o = np.ones((obj_h, obj_w), dtype=np.uint8) * occ_color

            # put occlusion inside the img
            in_img[obj_y:obj_y + obj_h, obj_x:obj_x + obj_w] = o

        return in_img

    def addNoise(self, in_img, in_label):
        """
        Add all possible noise to the image
        :param in_img: input image
        :param in_label: pupil location
        :return: return augmented image
        """
        # first make a copy of image and labels
        c_img = np.array(in_img, copy=True)
        c_label = np.array(in_label, copy=True)

        # apply noise
        c_img, c_label = self.downscale(c_img, c_label)
        c_img = self.addReflection(c_img)
        c_img = self.addOcclusion(c_img, c_label)
        c_img = self.addBlur(c_img)
        return c_img, c_label


if __name__ == "__main__":
    image_fn = "0in.jpg"
    img = cv2.imread(image_fn, 0)
    xml_path = "0gt.xml"
    e = ElementTree.parse(xml_path).getroot()
    x = np.round(np.float32(e[0].text))
    y = np.round(np.float32(e[1].text))
    w = np.round(np.float32(e[2].text))
    h = np.round(np.float32(e[3].text))
    a = np.round(np.float32(e[4].text))
    true_label = [x, y, w, h]

    ag = Augmentor('noisy_videos/', config)
    scaled_img, scaled_label = ag.addNoise(img, true_label)

    pil_img = Image.fromarray(anotator(scaled_img, scaled_label))
    pil_img.show()

    pil_img = Image.fromarray(anotator(img, true_label))
    pil_img.show()
    print("true label {}".format(true_label))
    print("scaled label {}".format(scaled_label))
