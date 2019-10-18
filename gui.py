import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import imgclasif
import imgds


class Application(ttk.Frame):
    """docstring for Application."""

    def __init__(self, master=None):
        super(Application, self).__init__(master)
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
        self.canvas_img.bind('<Button-1>', self.__click_add_class)
        self.canvas_img.bind('<Button-3>', self.__click_add_x)
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
        imgclasif.validate(
            self.pix_samples, self.pix_classes, amount, imgclasif.ClassifMethod(cm),
            imgclasif.EvalMethod(em))

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


def run_app():
    root = tk.Tk()
    root.title('Validación de Modelos')
    a = Application(root)
    root.mainloop()
