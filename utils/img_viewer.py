"""
Image viewer and optional re-saver for images generated during the runs.
"""
import os
import shutil
import argparse
from tkinter import *
from PIL import ImageTk, Image


class ImageViewer(object):

    def __init__(self, root, folder, save_folder, resave_size=None):
        self.root = root
        self._folder = folder
        self._run_folders = self.get_subfolders(folder)
        self._current_run = 0
        #print(self._run_folders)
        self._agent_folders = self.get_subfolders(self._run_folders[0])
        #print(self._agent_folders)
        self._current_folder = self._folder if len(self._agent_folders) == 0 else self._agent_folders[0]
        self._images, self._functions = self.parse_images(self._current_folder)
        #print(self._current_folder, self._images, self._functions)
        self._img_index = 0
        self._save_folder = save_folder
        self._resave_size = resave_size
        # Override closing protocol so that running tasks are cancelled before
        # exiting.
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        # Create base frame
        self.frame = Frame(root, borderwidth=0)
        self.frame.grid(row=0, column=0)
        # Create canvas for game of life cells
        self.image_frame = Frame(self.frame, borderwidth=0)
        self.image_frame.grid(sticky=N+E+S+W)

        self._image = Image.open(os.path.join(self._current_folder, self._images[self._img_index]))
        self._image = self._image.resize((400, 400))
        self.current_image = ImageTk.PhotoImage(self._image)
        self.image_label = Label(self.image_frame, image=self.current_image)
        self.image_label.grid(row=0, column=0)
        self.image_label.pack()
        img_name = self.format_image_name(self._images[self._img_index])
        self.image_name_label = Label(self.image_frame, text=img_name)
        self.image_name_label.grid(row=1, column=0)
        self.image_name_label.pack()

        self.gui_frame = Frame(self.frame,
                               borderwidth=0,
                               width=200)
        self.gui_frame.grid(row=1, column=0, sticky=N+E+S+W)
        self.button_frame = Frame(self.gui_frame,
                                  borderwidth=0)
        self.button_frame.grid(row=0, column=0, sticky=W)
        #self.prev_agent_button = Button(self.button_frame, text="Prev Agent ",
        #                          command=self.prev_agent)
        #self.prev_agent_button.grid(column=0, row=0, padx=2, pady=0, ipady=5, ipadx=5)
        self.prev_image_button = Button(self.button_frame, text=" <- ",
                                  command=self.prev_image)
        self.prev_image_button.grid(column=0, row=0, padx=2, pady=0, ipady=5, ipadx=5)
        self.save_button = Button(self.button_frame, text="Save",
                                   command=self.save_image)
        self.save_button.grid(column=1, row=0, padx=2, pady=0, ipady=5, ipadx=5)
        self.next_image_button = Button(self.button_frame, text=" -> ",
                                 command=self.next_image)
        self.next_image_button.grid(column=2, row=0, padx=2, pady=0, ipady=5, ipadx=5)
        self.next_agent_button = Button(self.button_frame, text="Next Agent",
                                 command=self.next_agent)
        self.next_agent_button.grid(column=3, row=0, padx=2, pady=0, ipady=5, ipadx=5)
        self.next_run_button = Button(self.button_frame, text="Next Run",
                                 command=self.next_run)
        self.next_run_button.grid(column=4, row=0, padx=2, pady=0, ipady=5, ipadx=5)

        self.root.bind('<Left>', self.prev_image)
        self.root.bind('<Right>', self.next_image)
        self.root.bind('<Up>', self.save_image)
        self.root.bind('<Return>', self.save_image)

    def save_image(self, event=None):
        img_path = os.path.join(self._current_folder, self._images[self._img_index])
        print("Saving image {} to {}.".format(self._images[self._img_index], self._save_folder))
        shutil.copy(img_path, self._save_folder)
        shutil.copy(os.path.join(self._current_folder, self._functions[self._img_index][1]), self._save_folder)

    def format_image_name(self, img_name):
        def get_aesthetic(aest):
            if aest == 'factor':
                return "GCF"
            if aest == 'entropy':
                return "ENT"
            if aest == 'symm':
                return "SYM"
            if aest == 'benford':
                return "BLW"
            if aest == 'aesthetics':
                return "FRD"

        folder_name = os.path.split(self._current_folder)[-1].split("_")
        agent_name = "R{:0>4} A{}:{}".format(self._current_run + 1, folder_name[-2], folder_name[-1])

        if len(img_name.split("-")) == 2:
            a1, a2 = img_name.split("-")
            a1 = a1.split("_")
            a2 = a2.split("_")
            aest1 = get_aesthetic(a1[-3])
            aest2 = get_aesthetic(a2[-1][:-4])
            return "{} {} {} - {}".format(agent_name, a1[1][3:-3], aest1, aest2)
        else:
            a = img_name.split("_")
            aest = get_aesthetic(a[-3])
            return "{} {} {}".format(agent_name, a[1][3:-3], aest)

    def show_image(self, img_index):
        self._image = Image.open(
            os.path.join(self._current_folder, self._images[self._img_index]))
        self._image = self._image.resize((400, 400))
        self.current_image = ImageTk.PhotoImage(self._image)
        #self.image_label.image = self.current_image
        self.image_label.destroy()
        self.image_label = Label(self.image_frame, image=self.current_image)
        self.image_label.grid(row=0, column=0)
        self.image_label.pack()
        self.image_name_label.destroy()
        img_name = self.format_image_name(self._images[self._img_index])
        self.image_name_label = Label(self.image_frame, text=img_name)
        self.image_name_label.grid(row=1, column=0)
        self.image_name_label.pack()
        #self.image_label.text = os.path.join(self._current_folder, self._images[self._img_index])
        #self.image_label.pack()

    def next_image(self, event=None):
        if self._img_index < len(self._images) - 1:
            self._img_index += 1
            self.show_image(self._img_index)
        else:
            self.next_agent()
        print("Next image! (index: {})".format(self._img_index))

    def prev_image(self, event=None):
        if self._img_index > 0:
            self._img_index -= 1
            self.show_image(self._img_index)
        else:
            self.prev_agent()
        print("Prev image! (index: {})".format(self._img_index))

    def next_run(self, event=None):
        self._current_run += 1
        if self._current_run == len(self._run_folders):
            pass
        else:
            self._agent_folders = self.get_subfolders(self._run_folders[self._current_run])
            self._current_folder = self._agent_folders[0]
            self._images, self._functions = self.parse_images(
                self._current_folder)
            # print(self._current_folder, self._images, self._functions)
            self._img_index = 0
            self.show_image(self._img_index)
            print("Changed to run: {}".format(self._run_folders[self._current_run]))

    def prev_run(self, event=None):
        # TODO: Implement this
        print("Not implemented!")
        pass

    def next_agent(self, event=None):
        self._img_index = 0
        idx = self._agent_folders.index(self._current_folder)
        if idx is not len(self._agent_folders) - 1:
            self._current_folder = self._agent_folders[idx + 1]
            self._images, self._functions = self.parse_images(
                self._current_folder)
            # print(self._current_folder, self._images, self._functions)
            self._img_index = 0
            self.show_image(self._img_index)
            print("Changed to agent: {}".format(self._current_folder))
        else:
            self.next_run()

    def prev_agent(self):
        # TODO: Implement this
        print("Not implemented!")
        pass

    def get_subfolders(self, folder):
        #print(folder)
        #print(os.listdir(folder))
        dirs = [os.path.join(folder, file) for file in os.listdir(folder) if
                os.path.isdir(os.path.join(folder, file))]
        return dirs


    def parse_images(self, folder):
        d = os.listdir(folder)
        files = [e for e in d if
                 os.path.isfile(os.path.join(folder, e))]
        #print(files)
        images = []
        functions = []
        for f in files:
            #print(f)
            fpath = os.path.join(folder, f)
            if f[-4:] == '.png':
                images.append(f)
                func_file = "f{}.txt".format(f[2:-4])
                with open(os.path.join(folder, func_file), 'r') as func:
                    fstr = func.read()
                functions.append((fstr, func_file))
        return images, functions

    def close(self, event=None):
        """Close GUI and exit.
        """
        #self.stop()
        self.root.withdraw()
        sys.exit()


if __name__ == "__main__":
    desc = """Command line script to view and re-save images generated in
    collaboration runs to a gallery folder.
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-f', metavar='run folder', type=str, dest='run_folder',
                        help="The run folder.")
    parser.add_argument('-s', metavar='save folder', type=str, dest='save_folder',
                        help="The save folder.", default="gallery")
    parser.add_argument('-z', metavar='size', type=int, dest='resave_size',
                        help="The size to re-save images.", default=None)

    args = parser.parse_args()
    root = Tk()
    root.resizable(False, False)
    root.title("Collaboration GP Image Viewer")
    gui = ImageViewer(root, args.run_folder, args.save_folder, args.resave_size)
    root.mainloop()
