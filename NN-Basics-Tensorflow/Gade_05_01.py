# Gade, Aniket
# 1001-505-046
# 2018-11-26
# Assignment-05-01
import Gade_05_02
import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.backends.tkagg as tkagg
import sklearn.datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class MainWindow(tk.Tk):
    """
    This class creates and controls the main window frames and widgets
    """

    def __init__(self, debug_print_flag=False):
        tk.Tk.__init__(self)
        self.debug_print_flag = debug_print_flag
        self.master_frame = tk.Frame(self)
        self.master_frame.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.rowconfigure(0, weight=1, minsize=500)
        self.columnconfigure(0, weight=1, minsize=500)
        self.master_frame.rowconfigure(2, weight=10, minsize=400, uniform='xx')
        self.master_frame.rowconfigure(3, weight=1, minsize=10, uniform='xx')
        self.master_frame.columnconfigure(0, weight=1, minsize=200, uniform='xx')
        self.master_frame.columnconfigure(1, weight=1, minsize=200, uniform='xx')
        # create all the widgets
        self.menu_bar = MenuBar(self, self.master_frame, background='orange')
        self.tool_bar = ToolBar(self, self.master_frame)
        self.left_frame = tk.Frame(self.master_frame)
        self.right_frame = tk.Frame(self.master_frame)
        self.status_bar = StatusBar(self, self.master_frame, bd=1, relief=tk.SUNKEN)
        # Arrange the widgets
        self.menu_bar.grid(row=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.tool_bar.grid(row=1, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.left_frame.grid(row=2, columnspan = 1 ,column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.right_frame.grid(row=2, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.status_bar.grid(row=3, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        # Create an object for plotting graphs in the left frame
        self.display_activation_functions = LeftFrame(self, self.left_frame, debug_print_flag=self.debug_print_flag)

class MenuBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        self.menu = tk.Menu(self.root)
        root.config(menu=self.menu)
        self.file_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="New", command=self.menu_callback)
        self.file_menu.add_command(label="Open...", command=self.menu_callback)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.menu_callback)
        self.dummy_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Dummy", menu=self.dummy_menu)
        self.dummy_menu.add_command(label="Item1", command=self.menu_item1_callback)
        self.dummy_menu.add_command(label="Item2", command=self.menu_item2_callback)
        self.help_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="About...", command=self.menu_help_callback)

    def menu_callback(self):
        self.root.status_bar.set('%s', "called the menu callback!")

    def menu_help_callback(self):
        self.root.status_bar.set('%s', "called the help menu callback!")

    def menu_item1_callback(self):
        self.root.status_bar.set('%s', "called item1 callback!")

    def menu_item2_callback(self):
        self.root.status_bar.set('%s', "called item2 callback!")


class ToolBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        self.master = master
        self.var_filename = tk.StringVar()
        self.var_filename.set('')
        self.ask_for_string = tk.Button(self, text="Ask for a string", command=self.ask_for_string)
        self.ask_for_string.grid(row=0, column=1)
        self.file_dialog_button = tk.Button(self, text="Open File Dialog", fg="blue", command=self.browse_file)
        self.file_dialog_button.grid(row=0, column=2)
        self.open_dialog_button = tk.Button(self, text="Open Dialog", fg="blue", command=self.open_dialog_callback)
        self.open_dialog_button.grid(row=0, column=3)

    def say_hi(self):
        self.root.status_bar.set('%s', "hi there, everyone!")

    def ask_for_string(self):
        s = simpledialog.askstring('My Dialog', 'Please enter a string')
        self.root.status_bar.set('%s', s)

    def ask_for_float(self):
        f = float(simpledialog.askfloat('My Dialog', 'Please enter a float'))
        self.root.status_bar.set('%s', str(f))

    def browse_file(self):
        self.var_filename.set(tk.filedialog.askopenfilename(filetypes=[("allfiles", "*"), ("pythonfiles", "*.txt")]))
        filename = self.var_filename.get()
        self.root.status_bar.set('%s', filename)

    def open_dialog_callback(self):
        d = MyDialog(self.root)
        self.root.status_bar.set('%s', "mydialog_callback pressed. Returned results: " + str(d.result))

    def button2_callback(self):
        self.root.status_bar.set('%s', 'button2 pressed.')

    def toolbar_draw_callback(self):
        self.root.display_graphics.create_graphic_objects()
        self.root.status_bar.set('%s', "called the draw callback!")

    def toolbar_callback(self):
        self.root.status_bar.set('%s', "called the toolbar callback!")


class MyDialog(tk.simpledialog.Dialog):
    def body(self, parent):
        tk.Label(parent, text="Integer:").grid(row=0, sticky=tk.W)
        tk.Label(parent, text="Float:").grid(row=1, column=0, sticky=tk.W)
        tk.Label(parent, text="String:").grid(row=1, column=2, sticky=tk.W)
        self.e1 = tk.Entry(parent)
        self.e1.insert(0, 0)
        self.e2 = tk.Entry(parent)
        self.e2.insert(0, 4.2)
        self.e3 = tk.Entry(parent)
        self.e3.insert(0, 'Default text')
        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=1, column=3)
        self.cb = tk.Checkbutton(parent, text="Hardcopy")
        self.cb.grid(row=3, columnspan=2, sticky=tk.W)

    def apply(self):
        try:
            first = int(self.e1.get())
            second = float(self.e2.get())
            third = self.e3.get()
            self.result = first, second, third
        except ValueError:
            tk.tkMessageBox.showwarning("Bad input", "Illegal values, please try again")


class StatusBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.label = tk.Label(self)
        self.label.grid(row=0, sticky=tk.N + tk.E + tk.S + tk.W)

    def set(self, format, *args):
        self.label.config(text=format % args)
        self.label.update_idletasks()

    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()


class LeftFrame:
    """
    This class creates and controls the widgets and figures in the left frame which
    are used to display the activation functions.
    """

    def __init__(self, root, master, debug_print_flag=False):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.alpha = 0.1
        self.lambda_ = 0.01
        self.bias = 0.0
        self.activation_type = "Relu"
        self.data_type = "s_curve"
        self.nodes = 100
        self.samples = 200
        self.classes = 4
        self.activation = 0
        self.data_points = None
        self.labels = None
        self.x_val = None
        self.y_val = None
        self.predictions = None
        self.points, self.labels = generate_data(self.data_type, self.samples, self.classes)
        #########################################################################
        #  Set up the plotting frame and controls frame
        #########################################################################
        master.rowconfigure(0, weight=10, minsize=200)
        master.columnconfigure(0, weight=1)
#        self.plot_frame = tk.Frame(self.master, borderwidth=10, relief=tk.SUNKEN)
#        self.plot_frame.grid(row=0, column=0, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
#        self.figure = plt.figure("")
#        self.axes = self.figure.add_axes([0.15, 0.15, 0.6, 0.8])
#        # self.axes = self.figure.add_axes()
#        self.axes = self.figure.gca()
#        self.axes.set_xlabel('Input')
#        self.axes.set_ylabel('Output')
#        # self.axes.margins(0.5)
#        self.axes.set_title("")
#        plt.xlim(self.xmin, self.xmax)
#        plt.ylim(self.ymin, self.ymax)
#        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
#        self.plot_widget = self.canvas.get_tk_widget()
#        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        # Create a frame to contain all the controls such as sliders, buttons, ...
        self.controls_frame = tk.Frame(self.master) 
        self.controls_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        #______________________________________________________
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, sticky=tk.N + tk.S)
        self.plot_frame.rowconfigure(0)
        self.plot_frame.columnconfigure(0)
        self.figure, self.axes_array = plt.subplots(1, 2)
        self.figure.set_size_inches(10, 6)
        self.axes = self.figure.gca()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        
        #______________________________________________________
        
        
#       -------------------------------------------  Button for Traninig  ---------------------------------------------------------------------------
        self.adj_weights_button = tk.Button(self.controls_frame, text="Train", fg="red", width=16, command=self.adj_weights_callback, height = 3)
        self.adj_weights_button.grid(row=0, column=8)
        
#        ------------------------------------------  Button for Creating Random Data  ---------------------------------------------------------------
        self.random_data_button = tk.Button(self.controls_frame, text="Randomize Weights", fg="red", width=16, command=self.random_data_callback, height = 3)
        self.random_data_button.grid(row=0, column=9)
#       --------------------------------------------------------------------------------------------------------------------------------------------
        
        #########################################################################
        #  Set up the control widgets such as sliders and selection boxes
        #########################################################################
#        --------------------------------------Slider for Alpha -----------------------------------------------------------
        self.alpha_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0.000, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="alpha",
                                            command=lambda event: self.alpha_slider_callback())
        self.alpha_slider.set(self.alpha)
        self.alpha_slider.bind("<ButtonRelease-1>", lambda event: self.alpha_slider_callback())
        self.alpha_slider.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)        
#---------------------------------------------------------------------------------------------------------------------------

#        --------------------------------------Slider for lambda -----------------------------------------------------------        
        self.lambda_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0.0, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Lambda",
                                            command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.lambda_)
        self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
        self.lambda_slider.grid(row=0, column=1, sticky=tk.N + tk.E + tk.S + tk.W)
        
#---------------------------------------------------------------------------------------------------------------------------

#        --------------------------------------Slider for classes -----------------------------------------------------------        
        self.class_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=2.0, to_=10.0, resolution=1.00, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Classes",
                                            command=lambda event: self.class_slider_callback())
        self.class_slider.set(self.classes)
        self.class_slider.bind("<ButtonRelease-1>", lambda event: self.class_slider_callback())
        self.class_slider.grid(row=0, column=5, sticky=tk.N + tk.E + tk.S + tk.W)
        
#---------------------------------------------------------------------------------------------------------------------------


#        --------------------------------------Slider for No. of Nodes -----------------------------------------------------------        
        self.nodes_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=1.0, to_=500.0, resolution=1.0, bg="#DDDDDD",
                                            activebackground="#FF0000", highlightcolor="#00FFFF", label="Nodes",
                                            command=lambda event: self.nodes_slider_callback())
        self.nodes_slider.set(self.nodes)
        self.nodes_slider.bind("<ButtonRelease-1>", lambda event: self.nodes_slider_callback())
        self.nodes_slider.grid(row=0, column=2, sticky=tk.N + tk.E + tk.S + tk.W)
        
#        -------------------------------------------------------------------------------------------------------------------
        self.samples_slider = tk.Scale(self.controls_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL, from_=4.0,
                                    to_=1000.0, resolution=1.0, bg="#DDDDDD", activebackground="#FF0000",
                                    highlightcolor="#00FFFF", label="Samples",
                                    command=lambda event: self.samples_slider_callback())
        self.samples_slider.set(self.samples)
        self.samples_slider.bind("<ButtonRelease-1>", lambda event: self.samples_slider_callback())
        self.samples_slider.grid(row=0, column=4, sticky=tk.N + tk.E + tk.S + tk.W)
        #########################################################################
        #  Set up the frame for drop down selection
        #########################################################################
#        self.label_for_activation_function = tk.Label(self.controls_frame, text="Activation:",
#                                                      justify="center")
#        self.label_for_activation_function.grid(row=0, column=3, sticky=tk.N + tk.E + tk.S + tk.W)
        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.controls_frame, self.activation_function_variable,
                                                          "Relu", "Sigmoid", command=lambda
                event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Relu")
        self.activation_function_dropdown.grid(row=0, column=6, sticky=tk.N + tk.E + tk.S + tk.W)
        
#        self.label_for_data_type = tk.Label(self.controls_frame, text="Data Type:",
#                                                      justify="center")
#        self.label_for_data_type.grid(row=0, column=5, sticky=tk.N + tk.E + tk.S + tk.W)
        self.data_type_variable = tk.StringVar()
        self.data_type_dropdown = tk.OptionMenu(self.controls_frame, self.data_type_variable,
                                                          "swiss_roll", "moons", "blobs", "s_curve",command=lambda
                event: self.data_type_dropdown_callback())
        self.data_type_variable.set("s_curve")
        self.data_type_dropdown.grid(row=0, column=7, sticky=tk.N + tk.E + tk.S + tk.W)
        
        self.canvas.get_tk_widget().bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.canvas.get_tk_widget().bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.left_mouse_release_callback)
        self.canvas.get_tk_widget().bind("<B1-Motion>", self.left_mouse_down_motion_callback)
        self.canvas.get_tk_widget().bind("<ButtonPress-3>", self.right_mouse_click_callback)
        self.canvas.get_tk_widget().bind("<ButtonRelease-3>", self.right_mouse_release_callback)
        self.canvas.get_tk_widget().bind("<B3-Motion>", self.right_mouse_down_motion_callback)
        self.canvas.get_tk_widget().bind("<Key>", self.key_pressed_callback)
        self.canvas.get_tk_widget().bind("<Up>", self.up_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Down>", self.down_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Right>", self.right_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Left>", self.left_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Shift-Up>", self.shift_up_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Shift-Down>", self.shift_down_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Shift-Right>", self.shift_right_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("<Shift-Left>", self.shift_left_arrow_pressed_callback)
        self.canvas.get_tk_widget().bind("f", self.f_key_pressed_callback)
        self.canvas.get_tk_widget().bind("b", self.b_key_pressed_callback)

    def key_pressed_callback(self, event):
        self.root.status_bar.set('%s', 'Key pressed')

    def up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Up arrow was pressed")

    def down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Down arrow was pressed")

    def right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Right arrow was pressed")

    def left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Left arrow was pressed")

    def shift_up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift up arrow was pressed")

    def shift_down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift down arrow was pressed")

    def shift_right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift right arrow was pressed")

    def shift_left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift left arrow was pressed")

    def f_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "f key was pressed")

    def b_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "b key was pressed")

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + 'x=' + str(event.x) + '   y=' + str(
            event.y))
        self.x = event.x
        self.y = event.y
        self.canvas.focus_set()

    def left_mouse_release_callback(self, event):
        self.root.status_bar.set('%s',
                                 'Left mouse button was released. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def left_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_release_callback(self, event):
        self.root.status_bar.set('%s',
                                 'Right mouse button was released. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def right_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + 'x=' + str(event.x) + '   y=' + str(
            event.y))
        self.x = event.x
        self.y = event.y

    # self.focus_set()
    def lambda_slider_callback(self):
        self.lambda_ = self.lambda_slider.get()
#        print(self.lambda_)
        
    def nodes_slider_callback(self):
        self.nodes = self.nodes_slider.get()
#        print(self.nodes)
        
    def class_slider_callback(self):
        self.classes = self.class_slider.get()
#        print(self.classes)
        
    def samples_slider_callback(self):
        self.samples = self.samples_slider.get()
#        print(self.samples)
        
    def alpha_slider_callback(self):
        self.alpha = self.alpha_slider.get()
#        print(self.alpha)

    def data_type_dropdown_callback(self):
        self.data_type = self.data_type_variable.get()
#        print(self.data_type)

    def activation_function_dropdown_callback(self):
        self.activation_type = self.activation_function_variable.get()
#        print(self.activation_type)
                
    def random_data_callback(self):
#        print("Randomize Weights")
        self.points, self.labels = generate_data(self.data_type, self.samples, self.classes)
        self.predictions = Gade_05_02.main(self.points, self.labels, self.classes, self.samples, self.nodes, self.activation_type, self.alpha, self.lambda_, random_weights = True)
        self.display()

    def adj_weights_callback(self):
        self.points, self.labels = generate_data(self.data_type, self.samples, self.classes)
        self.predictions = Gade_05_02.main(self.points, self.labels, self.classes, self.samples, self.nodes, self.activation_type, self.alpha, self.lambda_, random_weights = False)
        self.display()
        
    def display(self):
        self.axes.cla()
        self.axes_array[0].clear()
        self.axes_array[1].clear()
        temp = np.argmax(self.labels, axis=1)
        self.axes_array[0].scatter(self.points[:, 0], self.points[:, 1], c=temp, cmap=plt.cm.Accent)
        self.axes_array[0].set_title("Original Labels for "+str(self.data_type))
        
        self.axes_array[1].scatter(self.points[:, 0], self.points[:, 1], c=self.predictions, cmap=plt.cm.Accent)
        self.axes_array[1].set_title("Predications for "+str(self.data_type))
#        plt.scatter(self.points[:, 0], self.points[:, 1], c=self.predictions, cmap=plt.cm.Accent)
#        self.axes.xaxis.set_visible(True)
#        plt.title("Predications for "+str(self.data_type))
        self.canvas.draw()      
        
        
def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

def generate_data(dataset_name, n_samples, n_classes):
#    print("Dataset = ", dataset_name)
    if dataset_name == 'swiss_roll':
        data = sklearn.datasets.make_swiss_roll(n_samples, noise=1.5, random_state=99)[0]
        data = data[:, [0, 2]]
    if dataset_name == 'moons':
        data = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.15)[0]
    if dataset_name == 'blobs':
        data = sklearn.datasets.make_blobs(n_samples=n_samples, centers=n_classes*2, n_features=2, cluster_std=0.85*np.sqrt(n_classes), random_state=100)
        X,y = data[0]/10., [i % n_classes for i in data[1]]
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return(X, onehot_encoded)
    if dataset_name == 's_curve':
        data = sklearn.datasets.make_s_curve(n_samples=n_samples, noise=0.15, random_state=100)[0]
        data = data[:, [0,2]]/3.0

    ward = AgglomerativeClustering(n_clusters=n_classes*2, linkage='ward').fit(data)
    X,y =  data[:]+np.random.randn(*data.shape)*0.03, [i % n_classes for i in ward.labels_]
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#    return(X, y)
    return(X, onehot_encoded)

if __name__ == "__main__":
    main_window = MainWindow(debug_print_flag=False)
    # main_window.geometry("500x500")
    main_window.wm_state('zoomed')
    main_window.title('Assignment_05 --  Kamangar')
    main_window.minsize(800, 600)
    main_window.protocol("WM_DELETE_WINDOW", lambda root_window=main_window: close_window_callback(root_window))
    main_window.mainloop()