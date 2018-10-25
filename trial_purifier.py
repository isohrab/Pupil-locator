import glob
import os
import sys
import tkinter as tk
from tkinter import messagebox
from xml.etree import ElementTree

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from utils import annotator

FOLDER_PATH = 'purifier/folders.pkl'


def get_folders():
    """
    read all trial folders and return a dataframe
    :return: dataframe
    """
    # get the list of all folders
    folders_path = sorted(glob.glob("data/Original-data/belvedere/*"))

    # Create a dic to hold number of invalid images per folder
    f_dic = {}
    for path in folders_path:
        f_dic[path] = 0

    # Get the invalid images
    # invalid_images = glob.glob("data/Original-data/*/*/*.jpg_")

    # loop over all invalid images and +1 to the folder
    # for img in invalid_images:
    #     t = img.split("/")
    #     f_path = '/'.join(t[:-1])
    #     f_dic[f_path] += 1

    # make a data frame from dic
    # f_list = [[k, v] for k, v in f_dic.items()]

    folder_df = pd.DataFrame(data=folders_path, columns=["folder"])
    folder_df["checked"] = False
    #
    # folder_df = folder_df.sort_values(["invalids"], ascending=False)

    # folder_df.reset_index(inplace=True)

    folder_df.to_pickle(FOLDER_PATH)

    for i, row in folder_df.iterrows():
        print(row.folder)

    return folder_df


def get_dataframe(_path):
    """
    get a path and read images and labels (xmls) from current directory
    :param _path: directory path
    :return: a dataframe
    """
    all_images = sorted(glob.glob(_path + "/*.bmp"))
    all_xmls = sorted(glob.glob(_path + "/*.xml"))

    data = []
    for i, img in enumerate(all_images):
        vals = read_xml(all_xmls[i])

        # add image number to sort the dataframe based on it
        name = img.split("/")[-1]
        num = name.split(".")[0]
        num = int(num[:-2])
        data.append([img, vals[0], vals[1], vals[2], vals[3], vals[4], num])

    df = pd.DataFrame(data=data, columns=["path", "xt", "yt", "wt", "ht", "angt", "num"])

    df = df.sort_values(["num"])

    df.reset_index(inplace=True)

    df["status"] = 0


    return df


def read_xml(xml_path):
    e = ElementTree.parse(xml_path).getroot()
    x = np.float32(e[0].text)
    y = np.float32(e[1].text)
    w = np.float32(e[2].text)
    h = np.float32(e[3].text)
    a = np.float32(e[4].text)
    return [x, y, w, h, a]


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


class inspector_gui:

    def __init__(self, master, data):
        self.frame = tk.Frame(master)
        self.frame.pack_propagate(0)
        self.frame.pack(fill=tk.BOTH, expand=1)

        # Folder index
        self.f_idx = 0

        self.folder_df = data
        self.n_folders = len(data)

        self.current_df = None
        self.n_img = 0

        self.current_df_dirty = False

        # folder navigation
        self.prev_folder_btn = tk.Button(self.frame, text="previous Folder", command=lambda: self.change_folder(-1))
        self.prev_folder_btn.place(width=140, height=30, x=20, y=5)

        self.path_lbl = tk.Label(self.frame, text="Image path: ", anchor=tk.CENTER)
        self.path_lbl.place(width=380, height=20, x=200, y=5)

        self.next_folder_btn = tk.Button(self.frame, text="next Folder", command=lambda: self.change_folder(1))
        self.next_folder_btn.place(width=140, height=30, x=640, y=5)

        # big labeled image
        self.canvas = tk.Canvas(self.frame, width=576, height=576, bg="yellow")
        self.canvas.place(width=576, height=576, x=12, y=40)
        img = Image.open("0in.jpg")
        self.photo = ImageTk.PhotoImage(img)
        self.image_ref = self.canvas.create_image((288, 288), image=self.photo)

        # thumbsnail image
        self.canvas_s = tk.Canvas(self.frame, width=192, height=192)
        self.canvas_s.place(width=192, height=192, x=596, y=40)
        self.photo_s = ImageTk.PhotoImage(img)
        self.image_refs = self.canvas_s.create_image((96, 96), image=self.photo_s)

        self.pager_lbl = tk.Label(self.frame, text="0/1234", anchor=tk.CENTER)
        self.pager_lbl.place(width=192, height=20, x=596, y=225)

        self.status_lbl = tk.Label(self.frame, text="0", anchor=tk.CENTER, font=("Courier", 34))
        self.status_lbl.place(width=192, height=40, x=596, y=255)

        # true false buttons
        self.incorrect_btn = tk.Button(self.frame, text="Incorrect (i)", bg="red", command=lambda: self.updateDF(2))
        self.incorrect_btn.place(width=80, height=40, x=610, y=410)

        self.correct_btn = tk.Button(self.frame, text="correct (c)", bg="green", command=lambda: self.updateDF(1))
        self.correct_btn.place(width=80, height=40, x=700, y=410)

        # back and forward buttons for images
        self.backButton = tk.Button(self.frame, text="<- back", command=lambda: self.updateIndex(-1))
        self.backButton.place(width=80, height=30, x=610, y=470)

        self.nextButton = tk.Button(self.frame, text="next ->", command=lambda: self.updateIndex(1))
        self.nextButton.place(width=80, height=30, x=700, y=470)

        # capture image and save dataframe buttons
        self.capture_btn = tk.Button(self.frame, text="Capture (p)", command=self.capture)
        self.capture_btn.place(width=80, height=30, x=610, y=530)

        self.save_btn = tk.Button(self.frame, text="save", command=self.saveDF)
        self.save_btn.place(width=80, height=30, x=700, y=530)

        # export and rename buttons
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

        # select the first folder as start point
        self.goto_folder(0)


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
        status_0 = self.current_df.index[self.current_df["status"] == 0].tolist()
        status_0 = sorted(status_0)
        if len(status_0) == 0:
            status_1 = self.current_df.index[self.current_df["status"] == 1].tolist()
            status_1 = sorted(status_1)
            if len(status_1) == 0:
                return self.goto_folder(1)
            else:
                return status_1[0]
        else:
            return status_0[0]

    def capture(self):
        row = self.current_df.iloc[self.img_index]

        img = cv2.imread(row.path, cv2.IMREAD_GRAYSCALE)

        truth = [row.xt, row.yt, row.wt, row.ht, row.angt]

        # Update the labeled image
        img = annotator((0, 250, 0), img, *truth)  # Green

        save_path = row.path.replace("/", "-")
        cv2.imwrite("purifier/" + save_path, img)

    def change_folder(self, val):
        """
        update the folder index and clip it between 0 and n_folders
        :param val: +1 go next, -1 go previous
        :return: updated folder_idx
        """
        if self.current_df_dirty:
            res = messagebox.askquestion("Save Data", "Did you save the data?", icon='warning')
            if res == 'no':
                return

        self.f_idx += val
        self.f_idx = np.clip(self.f_idx, 0, self.n_folders - 1)

        self.goto_folder(self.f_idx)

    def goto_folder(self, idx):
        """
        Get the path from folder data frame. We should check if the upcomming folder
        has already a dataframe for its images. if not, create one.
        :param idx: index of current folder to be shown
        """
        # get the row of current path
        row = self.folder_df.iloc[idx]

        # check if dataframe is already exist
        df_name = row.folder.replace("/", "_")
        df_path = "purifier/" + df_name + ".pkl"

        if os.path.exists(df_path):
            self.current_df = pd.read_pickle(df_path)
        else:
            # read all images and labels in current directory
            self.current_df = get_dataframe(row.folder)

        # reset the image index
        self.img_index = self.findNextIndex()

        self.n_img = len(self.current_df)

        # update the folder name label
        new_text = "{0}".format(row.folder)
        self.path_lbl.configure(text=new_text)

        self.current_df_dirty = False

        # finally update GUI with new data
        self.updateGUI()

    def exportPath(self):
        """
        export path of images which flaged as incorrect
        :return:
        """
        incorrects = self.current_df[self.current_df.status == 2]

        path_txt = []
        # loop over rows and extract the paths
        for i, row in incorrects.iterrows():
            path_txt.append(row.path + "\n")

        # save file
        f_row = self.folder_df.iloc[self.f_idx]
        export_path = f_row.folder + "/incorrects.txt"
        open(export_path, mode='w').writelines(path_txt)

        # corrects = self.df[self.df.status == 1]
        # with open(CHECKED_PATH, mode='a') as f:
        #     for i, row in corrects.iterrows():
        #         path = row.trial + "/" + row.img_id + "\n"
        #         f.writelines(path)

        messagebox.showinfo("Export path", "incorrect paths exported successfuly at {}".format(export_path))

    def updateDF(self, val):
        """
        update the status of current row and go to next image
        :return:
        """
        self.current_df.at[self.img_index, "status"] = val
        r = self.current_df.iloc[self.img_index]
        print("{0} has been marked as {1}".format(r.path, r.status))
        self.current_df_dirty = True

        self.updateIndex(1)

    def saveDF(self):
        """
        save incorrect labeled images into a file
        :return:
        """
        # get the row of current path
        row = self.folder_df.iloc[self.f_idx]

        # check if dataframe is already exist
        df_name = row.folder.replace("/", "_")
        df_path = "purifier/"+df_name+".pkl"
        try:
            self.current_df.to_pickle(df_path)
        except IOError:
            print("IO Error")
        except RuntimeError:
            print("RuntimeError")
        except EOFError:
            print("EOFError")
        except OSError:
            print("OSError")
        except:
            print("Unexpected error:", sys.exc_info()[0])


        self.current_df_dirty = False
        messagebox.showinfo("save data", "Data saved successfuly at {}".format(df_path))

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
        row = self.current_df.iloc[self.img_index]

        # update pager
        new_text = "{0}/{1}".format(self.img_index + 1, self.n_img)
        self.pager_lbl.configure(text=new_text)

        # update status
        self.status_lbl.configure(text=str(row.status))

        # update image holder
        # load image
        file = row.path.split(".")[0]
        file = file + ".bmp"
        if row.status == 2:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        else:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        # update thumbnails before manipulation
        s_img = np.asarray(img, dtype=np.uint8)
        s_img = Image.fromarray(s_img, 'L')
        self.photo_s = ImageTk.PhotoImage(image=s_img)
        self.canvas_s.itemconfig(self.image_refs, image=self.photo_s)

        # resize image 3x and put label on it
        img = cv2.resize(img, (576, 576))
        truth = [row.xt * 3, row.yt * 3, row.wt * 3, row.ht * 3, row.angt]
        img = annotator((0, 250, 0), img, *truth)  # Green

        img = numpy2pil(img)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.itemconfig(self.image_ref, image=self.photo)

    def file_renamer(self):
        """
        get the file path of miss labeled data, and read the paths inside the file,
        and rename the extension part to jpg_ and xml_
        :param file_path: list of bad-labeled images
        :return:
        """
        f_row = self.folder_df.iloc[self.f_idx]
        export_path = f_row.folder + "/incorrects.txt"

        counter = 0
        with open(export_path, mode='r') as f:
            for line in f:
                line = line.strip()
                root = line.split(".")[0]
                os.rename(root + ".jpg", root + ".jpg_")

                xml = root.replace("in.", "gt.")

                os.rename(xml + ".xml", xml + ".xml_")
                counter += 1

        messagebox.showinfo("rename", " {} images has been renamed".format(counter))


if __name__ == '__main__':
    top = tk.Tk()
    top.title('Label inspector')
    top.geometry("800x620")
    top.resizable(0, 0)

    # check if folder dataframe already saved on disk
    if os.path.exists(FOLDER_PATH):
        fdf = pd.read_pickle(FOLDER_PATH)
    else:
        fdf = get_folders()

    inspector = inspector_gui(top, fdf)

    top.mainloop()
