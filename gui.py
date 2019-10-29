import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

import imgclasif
import imgds


class SupervisedApplication(ttk.Frame):
    """docstring for SupervisedApplication."""

    def __init__(self, master=None):
        super(SupervisedApplication, self).__init__(master)
        self.create_widgets()
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.TRUE)
        self.render = None
        self.img_file_name = None
        self.class_centers = []
        self.pix_samples = None
        self.pix_classes = None
        self.pix_coords = None
        self.x = None

    def create_widgets(self):
        # File selection
        #   Variables:
        self.var_img_path = tk.StringVar()
        #   Button: browse
        self.btn_browse = ttk.Button(
            self, text='Browse', command=self.__browse_image)
        self.btn_browse.grid(row=0, column=0, sticky=tk.W)
        #   Button: load
        self.btn_load = ttk.Button(
            self, text='Load', command=self.__load_image)
        self.btn_load.grid(row=0, column=1)
        #   Entry: file path
        self.entry_img_path = ttk.Entry(
            self, width=100, textvariable=self.var_img_path)
        self.entry_img_path.grid(row=0, column=2, sticky=tk.W + tk.E)

        # Algorithm selection
        #   Frame:
        self.frame_class = ttk.Frame(self)
        self.frame_class.grid(
            row=1, column=0, columnspan=2, sticky=tk.NW)
        #   Variables:
        self.var_class_method = tk.StringVar()
        self.var_eval_method = tk.StringVar()
        #   Label: Classification method
        self.lbl_class_method = tk.Label(
            self.frame_class, text='Classification Method:')
        self.lbl_class_method.grid(row=0, column=0, sticky=tk.W)
        #   Combo: Classification method selection
        self.combo_class_method = ttk.Combobox(
            self.frame_class, textvariable=self.var_class_method,
            state='readonly', values=self.__get_methods(
                imgclasif.ClassifMethod))
        self.combo_class_method.current(0)
        self.combo_class_method.grid(row=1, column=0)
        #   Label: Evaluation Method
        self.lbl_eval_method = tk.Label(
            self.frame_class, text='Evaluation Method:')
        self.lbl_eval_method.grid(row=2, column=0, sticky=tk.W)
        #   Combo: Evaluation method selection
        self.combo_eval_method = ttk.Combobox(
            self.frame_class, textvariable=self.var_eval_method,
            state='readonly',
            values=self.__get_methods(imgclasif.EvalMethod))
        self.combo_eval_method.current(0)
        self.combo_eval_method.grid(row=3, column=0)

        #   Label: Samples per class
        self.lbl_apc = ttk.Label(
            self.frame_class, text='Samples per class:')
        self.lbl_apc.grid(row=6, column=0, sticky=tk.W)
        #   Entry: Cantidad de clases
        self.var_apc = tk.IntVar()
        self.var_apc.set(300)
        self.entry_apc = ttk.Entry(
            self.frame_class, textvariable=self.var_apc, width=10,
            state='readonly ')
        self.entry_apc.grid(row=7, column=0)
        #   Button: Classify
        self.btn_classify = ttk.Button(
            self.frame_class, text='Classify', command=self.__classify)
        self.btn_classify.grid(row=9, column=0)
        #   Button: Evaluate
        self.btn_evaluate = ttk.Button(
            self.frame_class, text='Evaluate', command=self.__evaluate)
        self.btn_evaluate.grid(row=11, column=0)
        #   Button: Reset classes
        self.btn_reset = ttk.Button(
            self.frame_class, text='Reset', command=self.__reset)
        self.btn_reset.grid(row=13, column=0)

        self.frame_class.grid_rowconfigure(8, minsize=20)
        self.frame_class.grid_rowconfigure(10, minsize=20)
        self.frame_class.grid_rowconfigure(12, minsize=20)

        # Image display
        #   Canvas
        self.canvas_img = tk.Canvas(self)
        self.canvas_img.bind('<Button-1>', self.__click_add_x)
        self.canvas_img.bind('<Button-3>', self.__click_add_class)
        self.canvas_img.grid(row=1, column=2)

    def __get_methods(self, enum_methods):
        methods = []
        for cm in enum_methods:
            methods.append(cm.value)

        return methods

    def __browse_image(self):
        img_name = filedialog.askopenfilename(title='Select an image')
        self.var_img_path.set(img_name)

    def __load_image(self):
        fname = self.var_img_path.get()
        if not fname:
            messagebox.showinfo('Image', 'Please specify image path.')
            return

        self.canvas_img.delete('all')
        self.canvas_img['cursor'] = 'target'
        try:
            load = Image.open(fname)
        except FileNotFoundError:
            print('File', fname, 'doesn\'t exist.')
            return
        imgds.init(fname)
        self.img_file_name = fname

        self.class_centers.clear()
        self.render = ImageTk.PhotoImage(load)
        self.canvas_img.create_image(
            0, 0, image=self.render, anchor=tk.NW)
        self.canvas_img.config(width=load.size[0], height=load.size[1])

    def __classify(self):
        '''
        Classifies a point in the image according to previously selected
        classes.
        '''
        amount = self.var_apc.get()
        if not self.img_file_name:
            # TODO: Display a warning or something.
            return
        if amount <= 0:
            # TODO: Display a warning or something.
            return
        if self.class_centers is None:
            # TODO: Display a warning or something.
            return
        if self.x is None:
            # TODO: Display a warning or something.
            return
        if self.pix_samples is None or self.pix_classes is None:
            self.pix_samples, self.pix_classes, self.pix_coords = \
                imgds.get_class_samples(self.class_centers, amount)
            self.__repaint_image()
        cm = self.var_class_method.get()

        pred = imgclasif.classify(
            [imgds.get_sample(self.x)], self.pix_samples,
            self.pix_classes, imgclasif.ClassifMethod(cm))

        messagebox.showinfo(
            f'{cm} Result', f'Pixel {self.x} belongs to class {pred}')

    def __evaluate(self):
        '''
        Evaluate how well the selected algorithm classifies by showing a bar
        plot of the percentages each class got classified correctly.
        '''
        amount = self.var_apc.get()
        if not self.img_file_name or amount <= 0 or not self.class_centers:
            return

        if self.pix_samples is None or self.pix_classes is None:
            self.pix_samples, self.pix_classes, self.pix_coords = \
                imgds.get_class_samples(self.class_centers, amount)
            self.__repaint_image()

        cm = self.var_class_method.get()
        em = self.var_eval_method.get()
        imgclasif.validate(self.pix_samples, self.pix_classes, amount,
                           imgclasif.ClassifMethod(cm), imgclasif.EvalMethod(em))

    def __click_add_class(self, event):
        '''Adds the classes to the image in the canvas'''
        self.class_centers.append((event.x, event.y))

        wit = self.canvas_img.create_text(
            event.x, event.y, text=str(len(self.class_centers)), fill='black')
        wir = self.canvas_img.create_rectangle(self.canvas_img.bbox(wit),
                                               fill="wheat1")
        self.canvas_img.tag_lower(wir, wit)

    def __click_add_x(self, event):
        self.x = (event.x, event.y)
        self.__repaint_image()

    def __reset(self):
        self.class_centers.clear()
        self.pix_coords = None
        self.pix_samples = None
        self.pix_classes = None
        self.x = None

        self.canvas_img.delete('all')
        if self.render is not None:
            self.canvas_img.create_image(
                0, 0, image=self.render, anchor=tk.NW)

    def __repaint_image(self):
        self.canvas_img.delete('all')

        if self.render:
            self.canvas_img.create_image(
                0, 0, image=self.render, anchor=tk.NW)

        # Repaint samples
        if self.pix_coords is not None:
            for ps in self.pix_coords:
                self.canvas_img.create_oval(ps[1], ps[0], ps[1], ps[0],
                                            fill='dark slate gray')

        # Repaint classes
        for i, cl in enumerate(self.class_centers, 1):
            # self.canvas_img.create_text(*cl, text=str(i), fill='wheat1')
            wit = self.canvas_img.create_text(*cl, text=str(i), fill='black')
            wir = self.canvas_img.create_rectangle(self.canvas_img.bbox(wit),
                                                   fill="wheat1")
            self.canvas_img.tag_lower(wir, wit)

        # Repaint X
        if self.x:
            self.canvas_img.create_line(self.x[0] - 10, self.x[1],
                                        self.x[0] + 10, self.x[1],
                                        fill='dodger blue')
            self.canvas_img.create_line(self.x[0], self.x[1] - 10,
                                        self.x[0], self.x[1] + 10,
                                        fill='dodger blue')


class UnsupervisedApplication(ttk.Frame):
    """docstring for UnsupervisedApplication."""

    def __init__(self, master=None):
        super(UnsupervisedApplication, self).__init__(master)
        self.create_widgets()
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.TRUE)
        self.render = None
        self.img_file_name = None
        self.x = None
        self.pix_coords = []
        self.pix_samples = []
        self.k_centers = []
        self.k_labels = []

    def create_widgets(self):
        # File selection
        #   Variables:
        self.var_img_path = tk.StringVar()
        #   Button: browse
        self.btn_browse = ttk.Button(
            self, text='Browse', command=self.__browse_image)
        self.btn_browse.grid(row=0, column=0, sticky=tk.W)
        #   Button: load
        self.btn_load = ttk.Button(
            self, text='Load', command=self.__load_image)
        self.btn_load.grid(row=0, column=1)
        #   Entry: file path
        self.entry_img_path = ttk.Entry(
            self, width=100, textvariable=self.var_img_path)
        self.entry_img_path.grid(row=0, column=2, sticky=tk.W + tk.E)

        # Algorithm selection
        #   Frame:
        self.frame_class = ttk.Frame(self)
        self.frame_class.grid(
            row=1, column=0, columnspan=2, sticky=tk.NW)
        #   Variables:
        self.var_class_method = tk.StringVar()
        self.var_eval_method = tk.StringVar()
        #   Label: Classification method
        self.lbl_class_method = tk.Label(
            self.frame_class, text='Classification Method:')
        self.lbl_class_method.grid(row=0, column=0, sticky=tk.W)
        #   Combo: Classification method selection
        self.combo_class_method = ttk.Combobox(
            self.frame_class, textvariable=self.var_class_method,
            state='readonly', values=self.__get_methods(
                imgclasif.UnsupervisedMethod))
        self.combo_class_method.current(0)
        self.combo_class_method.grid(row=1, column=0)
        #   Label: Evaluation Method
        self.lbl_eval_method = tk.Label(
            self.frame_class, text='Evaluation Method:')
        self.lbl_eval_method.grid(row=2, column=0, sticky=tk.W)
        #   Combo: Evaluation method selection
        self.combo_eval_method = ttk.Combobox(
            self.frame_class, textvariable=self.var_eval_method,
            state='readonly',
            values=self.__get_methods(imgclasif.EvalMethod))
        self.combo_eval_method.current(0)
        self.combo_eval_method.grid(row=3, column=0)

        ##########
        # Inputs #
        ##########

        #   Label: Number of clusters
        self.lbl_nk = ttk.Label(self.frame_class, text='Clusters:')
        self.lbl_nk.grid(row=6, column=0, sticky=tk.W)
        #   Entry: Number of clusters
        self.var_nk = tk.IntVar()
        self.var_nk.set(3)
        self.entry_nk = ttk.Entry(
            self.frame_class, textvariable=self.var_nk, width=10,
            state='readonly ')
        self.entry_nk.grid(row=7, column=0)

        #   Label: Number of samples
        self.lbl_nsamples = ttk.Label(
            self.frame_class, text='Number of samples:')
        self.lbl_nsamples.grid(row=8, column=0, sticky=tk.W)
        #   Entry: Number of samples
        self.var_nsamples = tk.IntVar()
        self.var_nsamples.set(100)
        self.entry_nsamples = ttk.Entry(
            self.frame_class, textvariable=self.var_nsamples, width=10,
            state='readonly ')
        self.entry_nsamples.grid(row=9, column=0)

        #   Label: Threshold
        self.lbl_thres = ttk.Label(self.frame_class, text='Threshold:')
        self.lbl_thres.grid(row=10, column=0, sticky=tk.W)
        #   Entry: Threshold
        self.var_thres = tk.IntVar()
        self.var_thres.set(100)
        self.entry_thres = ttk.Entry(
            self.frame_class, textvariable=self.var_thres, width=10,
            state='readonly ')
        self.entry_thres.grid(row=11, column=0)

        ###########
        # Buttons #
        ###########
        #   Button: Get samples
        self.btn_sample = ttk.Button(
            self.frame_class, text='Sample', command=self.__random_samples)
        self.btn_sample.grid(row=13, column=0)
        #   Button: cluster
        self.btn_classify = ttk.Button(
            self.frame_class, text='Cluster data', command=self.__cluster)
        self.btn_classify.grid(row=14, column=0)
        #   Button: classify
        self.btn_classify = ttk.Button(
            self.frame_class, text='Classify', command=self.__classify)
        self.btn_classify.grid(row=15, column=0)
        #   Button: Evaluate
        self.btn_evaluate = ttk.Button(
            self.frame_class, text='Evaluate', command=self.__evaluate)
        self.btn_evaluate.grid(row=16, column=0)
        #   Button: Reset classes
        self.btn_reset = ttk.Button(
            self.frame_class, text='Reset', command=self.__reset)
        self.btn_reset.grid(row=17, column=0)

        self.frame_class.grid_rowconfigure(10, minsize=20)
        # self.frame_class.grid_rowconfigure(12, minsize=20)
        # self.frame_class.grid_rowconfigure(14, minsize=20)
        # self.frame_class.grid_rowconfigure(16, minsize=20)
        # self.frame_class.grid_rowconfigure(18, minsize=20)

        # Image display
        #   Canvas
        self.canvas_img = tk.Canvas(self)
        self.canvas_img.bind('<Button-1>', self.__click_add_x)
        self.canvas_img.bind('<Button-3>', self.__click_add_sample)
        self.canvas_img.grid(row=1, column=2)

    def __browse_image(self):
        img_name = filedialog.askopenfilename(title='Select an image')
        self.var_img_path.set(img_name)

    def __classify(self):
        pred = imgclasif.nearest_cluster([self.x], self.k_centers)
        messagebox.showinfo(
            'Classify', f'x belongs to class {self.k_labels[pred]}')

    def __cluster(self):
        '''
        Applies unsupervised classification to the dataset.
        '''
        if not self.img_file_name:
            messagebox.showerror('Classify', 'Please load an image first.')
            return

        if len(self.pix_samples) == 0:
            messagebox.showerror(
                'Classify', 'No data selected to form the clusters.')
            return

        cm = self.var_class_method.get()
        k = self.var_nk.get()
        t = self.var_thres.get()

        if cm == imgclasif.UnsupervisedMethod.CHA.value:
            labels = imgclasif.clusterize(
                self.pix_samples, method=imgclasif.UnsupervisedMethod(cm), thres=t)
        else:
            labels = imgclasif.clusterize(
                self.pix_samples, k=k, method=imgclasif.UnsupervisedMethod(cm))
        self.k_centers, self.k_labels = imgclasif.calculate_centers(
            np.array(self.pix_coords), labels)
        self.__repaint_image()

    def __click_add_x(self, event):
        if not self.render:
            return

        self.x = (event.x, event.y)
        self.__repaint_image()

    def __click_add_sample(self, event):
        # First validate
        if not self.render:
            return

        sample = imgds.get_sample((event.x, event.y))
        self.pix_coords.append((event.x, event.y))
        self.pix_samples.append(sample)
        self.__repaint_image()

    def __evaluate(self):
        '''
        Evaluate how well the selected algorithm classifies by showing a bar
        plot of the percentages each class got classified correctly.
        '''
        amount = self.var_apc.get()
        if not self.img_file_name or amount <= 0 or not self.class_centers:
            return

        if self.pix_samples is None or self.pix_classes is None:
            self.pix_samples, self.pix_classes, self.pix_coords = \
                imgds.get_class_samples(self.class_centers, amount)
            self.__repaint_image()

        cm = self.var_class_method.get()
        em = self.var_eval_method.get()
        imgclasif.validate(self.pix_samples, self.pix_classes,
                           amount, imgclasif.ClassifMethod(cm),
                           imgclasif.EvalMethod(em))

    def __get_methods(self, enum_methods):
        methods = []
        for cm in enum_methods:
            methods.append(cm.value)

        return methods

    def __load_image(self):
        fname = self.var_img_path.get()
        if not fname:
            messagebox.showinfo(message='Please specify image path.')
            return

        # Clean the canvas and stored data
        self.__reset()
        self.canvas_img.delete('all')
        self.canvas_img['cursor'] = 'target'
        try:
            load = Image.open(fname)
        except FileNotFoundError:
            print('File', fname, 'doesn\'t exist.')
            return
        imgds.init(fname)
        self.img_file_name = fname

        self.render = ImageTk.PhotoImage(load)
        self.canvas_img.create_image(0, 0, image=self.render, anchor=tk.NW)
        self.canvas_img.config(width=load.size[0], height=load.size[1])

    def __random_samples(self):
        '''
        Randomly samples some pixels from the image. Returns an
        array of tuples, which are the RGB values, and another
        array with with the coordinates from which they were
        sampled.


        __random_sampels() -> pixels, coordinates
        '''
        # First validate
        if not self.render:
            messagebox.showerror('Samples:', 'Image not loaded.')
            return
        try:
            amount = self.var_nsamples.get()
        except:
            messagebox.showerror('Samples:', 'Invalid input.')
            return

        if amount <= 0:
            messagebox.showerror('Samples:', 'Please specify image path.')
            return

        samples, coords = imgds.get_n_samples(amount)

        self.pix_samples.extend(samples)
        self.pix_coords.extend(coords)
        self.__repaint_image()

    def __repaint_image(self):
        self.canvas_img.delete('all')

        if self.render:
            self.canvas_img.create_image(
                0, 0, image=self.render, anchor=tk.NW)

        # Repaint samples
        if len(self.pix_coords) > 0:
            for ps in self.pix_coords:
                self.canvas_img.create_line(ps[0] - 10, ps[1],
                                            ps[0] + 10, ps[1],
                                            fill='dark slate gray')
                self.canvas_img.create_line(ps[0], ps[1] - 10,
                                            ps[0], ps[1] + 10,
                                            fill='dark slate gray')

        # Repaint cluster centers
        if len(self.k_centers) > 0:
            for i, cl in enumerate(self.k_centers, 0):
                wit = self.canvas_img.create_text(
                    *cl, text=str(self.k_labels[i]), fill='black')
                wir = self.canvas_img.create_rectangle(self.canvas_img.bbox(wit),
                                                       fill="wheat1")
                self.canvas_img.tag_lower(wir, wit)

        # Repaint X
        if self.x:
            self.canvas_img.create_line(self.x[0] - 10, self.x[1],
                                        self.x[0] + 10, self.x[1],
                                        fill='orange red')
            self.canvas_img.create_line(self.x[0], self.x[1] - 10,
                                        self.x[0], self.x[1] + 10,
                                        fill='orange red')

    def __reset(self):
        self.pix_coords.clear()
        self.pix_samples.clear()
        self.k_centers = []
        self.k_labels = []
        self.x = None

        self.canvas_img.delete('all')
        if self.render is not None:
            self.canvas_img.create_image(
                0, 0, image=self.render, anchor=tk.NW)


def run_app():
    root = tk.Tk()
    root.title('Validación de Modelos')
    nb = ttk.Notebook(root)
    sa = SupervisedApplication(root)
    ua = UnsupervisedApplication(root)
    nb.add(sa, text='Supervised')
    nb.add(ua, text='Unsupervised')
    nb.pack()
    root.mainloop()
