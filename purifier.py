import glob
import os
import tkinter as tk
from tkinter import messagebox
from xml.etree import ElementTree

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageTk

from config import config
from logger import Logger
from reporter import load_model
from utils import annotator
from utils import change_channel, gray_normalizer

DF3P_PATH = "purifier/df3p.pkl"
EXPORT_PATH = "purifier/incorrects.txt"
RESULTS_PATH = "purifier/results.txt"
CHECKED_PATH = "purifier/checked.txt"


def numpy2pil(np_array: np.ndarray) -> Image:
    """
    convert an HxWx3 numpy array into an RGB Image
    :param np_array: input numpy array
    :return: A PIL Image object
    """
    assert_mfg = "input shall be a HxWx3 ndarray"
    assert isinstance(np_array, np.ndarray), assert_mfg
    assert np.ndim(np_array) == 3, assert_mfg
    assert np_array.shape[2] == 3, assert_mfg
    img = Image.fromarray(np_array, 'RGB')
    return img


def read_xml(xml_path):
    e = ElementTree.parse(xml_path).getroot()
    x = np.float32(e[0].text)
    y = np.float32(e[1].text)
    w = np.float32(e[2].text)
    h = np.float32(e[3].text)
    a = np.float32(e[4].text)
    return [x, y, w, h, a]


def check_trials():
    logger = Logger("INC", "Inc_Purifier3", "", config, dir="models/")
    with tf.Session() as sess:
        # load best model
        model = load_model(sess, "INC", "Inc_Purifier3", logger)

        # print the result for different pixel error
        pixel_errors = [1, 2, 3, 4, 5, 7, 10, 15, 20]

        trials_path = sorted(glob.glob("data/Original-data/*/*"))

        results = []

        for i, path in enumerate(trials_path):
            print("{0:3} reading {1}".format(i, path))

            images_path = glob.glob(path + "/*.jpg")
            images = []
            truths = []
            img_paths = []

            for ii, img_path in enumerate(images_path):

                _xml_path = img_path.split(".")[0] + ".xml"
                _xml_path = _xml_path.replace("in.", "gt.")
                truth = read_xml(_xml_path)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = change_channel(img)
                img = gray_normalizer(img)

                images.append(img)
                truths.append(truth)
                img_paths.append(img_path)
                if len(images) == 64 or ii == (len(images_path) - 1):
                    pred = model.predict(sess, images)

                    for iii, p in enumerate(pred):
                        img_id = img_paths[iii]
                        img_id = img_id.split("/")[4]
                        img_id = img_id.split(".")[0]
                        # trial_path, img_id, xt, yt, wt, ht, at, xp, yp, wp, hp, ap
                        result = "{0};{1};{2};{3};{4};{5};{6};{7};{8};{9};{10};{11}\n".format(path,
                                                                                              img_id,
                                                                                              *truths[iii],
                                                                                              *pred[iii])
                        results.append(result)

                    images = []
                    truths = []
                    img_paths = []

        open(RESULTS_PATH, mode="w").writelines(results)


class inspector_gui:

    def __init__(self, master, data):
        self.frame = tk.Frame(master)
        self.frame.pack_propagate(0)
        self.frame.pack(fill=tk.BOTH, expand=1)

        self.df = data

        if "status" not in list(self.df):
            self.df["status"] = 0

        self.n_img = len(data)

        self.img_index = self.findNextIndex()

        self.path_lbl = tk.Label(self.frame, text="Image path: ", anchor=tk.NW)
        self.path_lbl.place(width=400, height=20, x=10, y=5)

        self.pager_lbl = tk.Label(self.frame, text="0/" + str(self.n_img), anchor=tk.NE)
        self.pager_lbl.place(width=120, height=20, x=470, y=5)

        # big labeled image
        self.canvas = tk.Canvas(self.frame, width=576, height=576, bg="yellow")
        self.canvas.place(width=576, height=576, x=12, y=25)
        img = Image.open("0in.jpg")
        self.photo = ImageTk.PhotoImage(img)
        self.image_ref = self.canvas.create_image((288, 288), image=self.photo)

        # thumbsnail image
        self.canvas_s = tk.Canvas(self.frame, width=192, height=192)
        self.canvas_s.place(width=192, height=192, x=596, y=25)
        self.photo_s = ImageTk.PhotoImage(img)
        self.image_refs = self.canvas_s.create_image((96, 96), image=self.photo_s)

        self.status_lbl = tk.Label(self.frame, text="0", anchor=tk.CENTER, font=("Courier", 24))
        self.status_lbl.place(width=192, height=30, x=596, y=210)

        # put the truth and predicted label
        self.header_lbl = tk.Label(self.frame, text="\t Truth  Prediction", anchor=tk.W)
        self.header_lbl.place(width=180, height=30, x=596, y=240)

        self.x_lbl = tk.Label(self.frame, text="x:", anchor=tk.W)
        self.x_lbl.place(width=180, height=30, x=596, y=270)

        self.y_lbl = tk.Label(self.frame, text="y:", anchor=tk.W)
        self.y_lbl.place(width=180, height=30, x=596, y=300)

        self.w_lbl = tk.Label(self.frame, text="w:", anchor=tk.W)
        self.w_lbl.place(width=180, height=30, x=596, y=330)

        self.h_lbl = tk.Label(self.frame, text="h:", anchor=tk.W)
        self.h_lbl.place(width=180, height=30, x=596, y=360)

        self.a_lbl = tk.Label(self.frame, text="a:", anchor=tk.W)
        self.a_lbl.place(width=180, height=30, x=596, y=390)

        self.incorrect_btn = tk.Button(self.frame, text="Incorrect (i)", bg="red", command=lambda: self.updateDF(2))
        self.incorrect_btn.place(width=80, height=40, x=610, y=430)

        self.correct_btn = tk.Button(self.frame, text="correct (c)", bg="green", command=lambda: self.updateDF(1))
        self.correct_btn.place(width=80, height=40, x=700, y=430)

        self.capture_btn = tk.Button(self.frame, text="Capture (p)", command=self.capture)
        self.capture_btn.place(width=80, height=30, x=610, y=480)

        self.save_btn = tk.Button(self.frame, text="save", command=self.saveDF)
        self.save_btn.place(width=80, height=30, x=700, y=480)

        self.backButton = tk.Button(self.frame, text="<- back", command=lambda: self.updateIndex(-1))
        self.backButton.place(width=80, height=30, x=610, y=530)

        self.nextButton = tk.Button(self.frame, text="next ->", command=lambda: self.updateIndex(1))
        self.nextButton.place(width=80, height=30, x=700, y=530)

        self.export_btn = tk.Button(self.frame, text="export path", command=self.exportPath)
        self.export_btn.place(width=80, height=30, x=700, y=580)

        self.rename_btn = tk.Button(self.frame, text="rename path", command=self.file_renamer)
        self.rename_btn.place(width=80, height=30, x=610, y=580)

        # bind events with keyboard
        master.bind('<Left>', self.leftKey)
        master.bind('<Right>', self.rightKey)
        master.bind('i', self.enterKey)
        master.bind('c', self.spaceKey)
        master.bind('p', self.captureKey)

        # show the first image
        self.updateGUI()

    def rightKey(self, event):
        self.updateIndex(1)

    def leftKey(self, event):
        self.updateIndex(-1)

    def spaceKey(self, event):
        self.updateDF(1)

    def enterKey(self, event):
        self.updateDF(2)

    def captureKey(self, event):
        self.capture()

    def findNextIndex(self):
        """
        loop over dataframe and return an index with status 0
        if not found, alert and return index= 0
        :return:
        """
        status_0 = self.df.index[self.df["status"] == 0].tolist()
        status_0 = sorted(status_0)
        if len(status_0) == 0:
            return 0
        else:
            return status_0[0]

    def capture(self):
        row = self.df.iloc[self.img_index]
        new_path = "{0}/{1}.jpg".format(row.trial, row.img_id)

        img = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)

        truth = [row.xt, row.yt, row.wt, row.ht, row.angt]
        pred = [row.xp, row.yp, row.wp, row.hp, row.angp]

        # Update the labeled image
        img = annotator((120, 120, 120), img, *pred)  # gray
        img = annotator((0, 250, 0), img, *truth)  # Green

        save_path = new_path.replace("/", "-")
        cv2.imwrite("purifier/" + save_path, img)

    def exportPath(self):
        """
        export path of images which flaged as incorrect
        :return:
        """
        incorrects = self.df[self.df.status == 2]

        path_txt = []
        # loop over rows and extract the paths
        for i, row in incorrects.iterrows():
            path = "{0}/{1}.jpg\n".format(row.trial, row.img_id)
            path_txt.append(path)

        # save file
        open(EXPORT_PATH, mode='w').writelines(path_txt)

        corrects = self.df[self.df.status == 1]
        with open(CHECKED_PATH, mode='a') as f:
            for i, row in corrects.iterrows():
                path = row.trial + "/" + row.img_id + "\n"
                f.writelines(path)

        messagebox.showinfo("Export path", "incorrect paths exported successfuly at {}".format(EXPORT_PATH))

    def updateDF(self, val):
        """
        update the status of current row and go to next image
        :return:
        """
        self.df.at[self.img_index, "status"] = val
        r = self.df.iloc[self.img_index]
        print("{0}/{1} has been marked as {2}".format(r.trial, r.img_id, r.status))
        self.updateIndex(1)

    def saveDF(self):
        """
        save incorrect labeled images into a file
        :return:
        """
        self.df.to_pickle(DF3P_PATH)
        messagebox.showinfo("save data", "Data saved successfuly at {}".format(DF3P_PATH))

    def updateIndex(self, val):
        """
        update the image index and clip between 0, len(n_img). finally update the GUI
        :param val:
        :return:
        """
        self.img_index += val
        self.img_index = np.clip(self.img_index, 0, self.n_img - 1)

        self.updateGUI()

    def updateGUI(self):
        """
        update the GUI based on img_index
        :return:
        """
        row = self.df.iloc[self.img_index]

        # update path
        new_path = "{0}/{1}.jpg".format(row.trial, row.img_id)

        # update path_lbl
        new_lbl = "Path: {0}/{1}.jpg".format(row.trial, row.img_id)
        self.path_lbl.configure(text=new_lbl)

        # update pager
        new_text = "{0}/{1}".format(self.img_index + 1, self.n_img)
        self.pager_lbl.configure(text=new_text)

        # update status
        self.status_lbl.configure(text=str(row.status))

        # update truth and predicted labels
        new_x = "x:\t{0:5.1f}  {1:5.1f}".format(row.xt, row.xp)
        self.x_lbl.configure(text=new_x)

        new_y = "y:\t{0:5.1f}  {1:5.1f}".format(row.yt, row.yp)
        self.y_lbl.configure(text=new_y)

        new_w = "w:\t{0:5.1f}  {1:5.1f}".format(row.wt, row.wp)
        self.w_lbl.configure(text=new_w)

        new_h = "h:\t{0:5.1f}  {1:5.1f}".format(row.ht, row.hp)
        self.h_lbl.configure(text=new_h)

        new_ang = "a:\t{0:5.1f}  {1:5.1f}".format(row.angt, row.angp)
        self.a_lbl.configure(text=new_ang)

        # update image holder
        # load image
        img = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
        truth = [row.xt, row.yt, row.wt, row.ht, row.angt]
        pred = [row.xp, row.yp, row.wp, row.hp, row.angp]

        # update thumbnails before manipulation
        s_img = np.asarray(img, dtype=np.uint8)
        s_img = Image.fromarray(s_img, 'L')
        self.photo_s = ImageTk.PhotoImage(image=s_img)
        self.canvas_s.itemconfig(self.image_refs, image=self.photo_s)

        # Update the labeled image
        img = annotator((120, 120, 120), img, *pred)  # gray
        img = annotator((0, 250, 0), img, *truth)  # Green

        img = numpy2pil(img)
        img = img.resize((576, 576), )
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(self.image_ref, image=self.photo)

    def file_renamer(self):
        """
        get the file path of miss labeled data, and read the paths inside the file,
        and rename the extension part to jpg_ and xml_
        :param file_path: list of bad-labeled images
        :return:
        """
        counter = 0
        with open(EXPORT_PATH, mode='r') as f:
            for line in f:
                line = line.strip()
                root = line.split(".")[0]
                os.rename(root + ".jpg", root + ".jpg_")

                xml1 = root + ".xml"
                xml1 = xml1.replace("in.", "gt.")

                xml2 = root + ".xml_"
                xml2 = xml2.replace("in.", "gt.")
                os.rename(xml1, xml2)
                counter += 1

        print("{0} images has been renamed".format(counter))


def calculate_diff(pix_error):
    # first check if results.csv is already produced
    if not os.path.exists("purifier/results.txt"):
        check_trials()

    # now calculate the difference and save it to disk
    df = pd.read_csv("purifier/results.txt", sep=";", names=["trial", "img_id",
                                                             "xt", "yt", "wt", "ht", "angt",
                                                             "xp", "yp", "wp", "hp", "angp"])

    # read the checked list
    checked = []
    with open(CHECKED_PATH, mode='r') as f:
        for line in f:
            line = line.strip()
            checked.append(line)

    dx = df.xt[:] - df.xp[:]
    dy = df.yt[:] - df.yp[:]
    dw = (df.wt[:] - df.wp[:]) * 0.5
    dh = (df.ht[:] - df.hp[:]) * 0.5
    # dang = (df.angt[:] - df.angp[:]) * 0.2

    diff = np.sqrt(dx * dx + dy * dy + dw * dw + dh * dh)
    diff3p = diff >= pix_error
    df["d3p"] = diff3p
    df3p = df[df.d3p == True]
    df3p["duplicate"] = False

    for i, row in df3p.iterrows():
        path = row.trial + "/" + row.img_id
        if path in checked:
            df3p.at[i, "duplicate"] = True

    df3p = df3p[df3p.duplicate == False]

    df3p = df3p.reset_index()
    df3p.to_pickle(DF3P_PATH)
    return df3p


if __name__ == '__main__':
    top = tk.Tk()
    top.title('Label inspector')
    top.geometry("800x620")
    top.resizable(0, 0)

    # check if a dataframe already saved on disk
    if os.path.exists(DF3P_PATH):
        df3p = pd.read_pickle(DF3P_PATH)
    else:
        df3p = calculate_diff(5)

    inspector = inspector_gui(top, df3p)

    top.mainloop()
