#!/usr/bin/env python3.9  ## <-- or replace with your path to Python<3.10

# TODO:
#
# * add animation
# * add multiple director glyph slices
# * add omega vector
# * check animation for open-Qmin data
# * add opacity controls
# * add boundary solid color controls
# * add clim, cmap controls
# * add auto-stitching for open-Qmin mpi data
# * add data stride handling (including scaling derivatives)
# * add ellipses as optional replacement for cylinders

# Known issues:
# * scalar bar text looks gross... is its antialiasing disabled?

import numpy as np
from pandas import read_csv, DataFrame
from qtpy import QtWidgets as qw
import qtpy.QtCore as Qt
import pyvista as pv
import pyvistaqt as pvqt
import sys
import glob

class ViewMinPlot(pvqt.BackgroundPlotter):
    def __init__(self, filenames=[], user_settings={}):
        super().__init__(#multi_samples=8,line_smoothing=True, point_smoothing=True, polygon_smoothing=True,
        )
        self.theme.antialiasing=True
        self.finished_setup = False
        self.make_empty_convenience_arrays()
        self.set_settings(user_settings)
        self.app_window.setWindowTitle("open-ViewMin nematic visualization environment")
        self.rescale_lights_intensity(16)
        self.setup_import_menu()
        self.renderer.add_axes(interactive=True, color='black') # xyz axes arrows
        self.renderer.set_background("white")
        self.enable_eye_dome_lighting()
        self.q0 = 0.

        if len(filenames) > 0:
            self.load(filenames)


    def load(self, filenames):
        if type(filenames) is str:
            filenames = [filenames]
        try:
            first_frame_data = np.asarray(self.fast_import(filenames[0]), dtype=float)
        except ValueError: # detect whether file is from old Qmin
            self.legacy_Qmin_import(filenames)
        else:
            self.data.append(first_frame_data)  # reads data files into self.data
            self.read(filenames[1:])
            self.get_data_from_first_file(self.data[0])
            for file_data in self.data:
                self.meshdata_from_file(file_data) # sets self.fullmesh and appends it to self.fullmeshes
            self.init_procedures_after_data_import()

    def make_empty_convenience_arrays(self):
        self.data = []
        self.fullmeshes = []
        self.widgets = {}
        self.slices = {}
        self.isosurfaces = {}
        self.visibility_checkboxes = {}
        self.QSliders={}
        self.QSliders_labels = {}
        self.QSliders_updaters = {}
        self.QSliders_input_boxes = {}
        self.toolbars = {}
        self.colors = {}
        self.refresh_functions = {}
        self.colorbars = {}
        self.elements = self.renderer.actors

    def init_procedures_after_data_import(self):
        if not self.finished_setup:
            self.add_mesh(self.fullmesh.outline(), color='black', name="outline") # bounding box
            self.frame_num = 0
            self.setup_animation_buttons()
            self.load_frame()
            self.setup_QSliders()
            self.setup_isosurfaces()
            self.setup_menus()
            self.finished_setup = True


    def scalar_fields(self):
        return [ array_name for array_name in self.fullmesh.array_names
                 if len(self.fullmesh[array_name].shape)==1 ]

    def read(self, filenames, **kwargs):
        for filename in filenames:
            self.data.append(np.array(read_csv(filename, sep="\t", header=None)))

    def get_data_from_first_file(self, file_data):
        self.Lx, self.Ly, self.Lz = [ int(item+1) for item in file_data[-1,:3] ]

    def dims(self):
        return (self.Lx, self.Ly, self.Lz)

    def load_frame(self, frame_num=None):
        if frame_num is not None:
            self.frame_num = frame_num
        if self.frame_num >= len(self.fullmeshes):
            self.frame_num = len(self.fullmeshes)-1
        elif self.frame_num < 0:
            self.frame_num = 0
        self.fullmesh = self.fullmeshes[self.frame_num]
        self.make_coarsemesh(self.director_resolution)
        for actor_name in self.refresh_functions.keys():
            self.refresh_functions[actor_name]()
        self.frame_spinbox.setValue(self.frame_num)


    def next_frame(self):
        self.frame_num += 1
        self.load_frame()

    def previous_frame(self):
        self.frame_num -= 1
        self.load_frame()

    def first_frame(self):
        self.frame_num = 0
        self.load_frame()

    def last_frame(self):
        self.frame_num = len(self.fullmeshes)-1
        self.load_frame()

    def setup_animation_buttons(self):
        btn_width=45
        self.animation_buttons_toolbar = qw.QToolBar("Animation")
        tb = self.animation_buttons_toolbar
        tb.setStyleSheet("QToolBar{spacing:0px;}")

        sb = qw.QSpinBox(minimum=0, maximum=len(self.fullmeshes))
        sb.setValue(self.frame_num)
        sb.editingFinished.connect(lambda : self.load_frame(frame_num=sb.value()))
        self.frame_spinbox = sb

        btn = qw.QPushButton('⏮')
        btn.setFixedWidth(btn_width)
        btn.released.connect(self.first_frame)
        btn.released.connect(lambda : sb.setValue(self.frame_num))
        tb.addWidget(btn)

        btn = qw.QPushButton('⬅️')
        btn.setFixedWidth(btn_width)
        btn.released.connect(self.previous_frame)
        btn.released.connect(lambda : sb.setValue(self.frame_num))
        tb.addWidget(btn)

        tb.addWidget(sb)

        btn = qw.QPushButton('➡️')
        btn.released.connect(self.next_frame)
        btn.released.connect(lambda : sb.setValue(self.frame_num))
        btn.setFixedWidth(btn_width)
        tb.addWidget(btn)

        btn = qw.QPushButton('⏭')
        btn.released.connect(self.last_frame)
        btn.released.connect(lambda : sb.setValue(self.frame_num))
        btn.setFixedWidth(btn_width)
        tb.addWidget(btn)

        self.app_window.addToolBar(tb)


    def setup_import_menu(self):
        self.the_import_menu = self.main_menu.addMenu('Import')
        self.the_import_menu.addAction(
            'Open file(s)...',
            self.open_files_dialog
        )

    def setup_menus(self):
        self.toggle_menu = self.main_menu.addMenu('Toggle')
        self.toggle_menu.aboutToShow.connect(self.update_toggle_menu)
        self.the_Add_menu = self.main_menu.addMenu("Add")
        self.widget_menu = self.main_menu.addMenu("Widgets")
        self.widget_menu.aboutToShow.connect(self.update_widget_menu)
        self.the_add_slice_menu = self.the_Add_menu.addMenu("Slice")
        def add_slice_menu_update():
            menu = self.the_add_slice_menu
            menu.clear()
            for array_name in sorted(self.scalar_fields()):
                menu_action = menu.addAction(
                    array_name,
                    self.add_slice_aux(array_name)
                )
                menu_action.setCheckable(False)
        self.the_add_slice_menu.aboutToShow.connect(add_slice_menu_update)
        self.the_add_isosurface_menu = self.the_Add_menu.addMenu("Isosurface")
        def add_isosurface_menu_update():
            menu = self.the_add_isosurface_menu
            menu.clear()
            for array_name in sorted(self.scalar_fields()):
                menu_action = menu.addAction(
                    array_name,
                    self.add_isosurface_slider_aux(array_name)
                )
                menu_action.setCheckable(False)
        self.the_add_isosurface_menu.aboutToShow.connect(add_isosurface_menu_update)
        self.add_plane_widget_to_widget_menu("director_slice_widget")
        self.widget_menu.triggered.connect(
            lambda: self.widget_menu.actions()[0].setChecked(
                self.widgets["director_slice_widget"].GetEnabled()
            )
        )
        self.the_color_by_menu = self.main_menu.addMenu("Color by")
        self.the_color_by_menu.aboutToShow.connect(self.color_by_menu_update)
        for scalar_bar_name in self.scalar_bars.keys():
            self.add_scalar_bar_to_toggle_menu(scalar_bar_name)

    def color_by_menu_update(self):
        menu = self.the_color_by_menu
        menu.clear()
        for actor_name in self.colors.keys(): #self.isosurfaces.keys():
            # exclude color bars and widget outlines
            if not "Addr=" in actor_name and not (
                    len(actor_name.split('outline'))==2
                    and len(actor_name.split('outline')[0])>0
            ):
                submenu = menu.addMenu(actor_name)
                submenu.aboutToShow.connect(self.color_by_submenu_update(actor_name, submenu))


    def make_color_by_submenu_callback(self, actor_name, scalar_field):
        def submenu_callback():
            self.set_color(actor_name, scalar_field)
        return submenu_callback

    def color_by_submenu_update(self, actor_name, submenu):
        def return_function():
            submenu.clear()
            for scalar_field in sorted(self.scalar_fields()):
                submenu.addAction(
                    scalar_field,
                    self.make_color_by_submenu_callback(actor_name, scalar_field)
                )
        return return_function

    def update_toggle_menu(self):
        self.toggle_menu.clear()
        for actor_name in self.renderer.actors.keys():
            # exclude color bars and widget outlines
            if (not "Addr=" in actor_name
                and not (len(actor_name.split('outline'))==2
                         and len(actor_name.split('outline')[0])>0
                        )
            ):
                self.add_to_toggle_menu(actor_name)
        for scalar_bar_name in self.scalar_bars.keys():
            self.add_scalar_bar_to_toggle_menu(scalar_bar_name)


    def update_widget_menu(self):
        self.widget_menu.clear()
        for widget_name in self.widgets.keys():
            self.add_plane_widget_to_widget_menu(widget_name)

    def add_to_toggle_menu(self, actor_name):
        menu_action = self.toggle_menu.addAction(actor_name,
                              self.generic_menu_toggle(actor_name))
        menu_action.setCheckable(True)
        is_visible = self.renderer.actors[actor_name].GetVisibility()
        menu_action.setChecked(is_visible)

    def add_scalar_bar_to_toggle_menu(self, scalar_bar_name):
        scalar_bar = self.scalar_bars[scalar_bar_name]
        menu_action = self.toggle_menu.addAction("└─ " + scalar_bar_name + " colorbar",
            lambda : self.toggle_visibility(scalar_bar)
        )
        menu_action.setCheckable(True)
        is_visible = self.scalar_bars[scalar_bar_name].GetVisibility()
        menu_action.setChecked(is_visible)


    def toggle_visibility(self, actor):
        if type(actor) is str:
            actor_name = actor
            actor = self.renderer.actors[actor_name]
            if actor_name in self.visibility_checkboxes.keys():
                self.visibility_checkboxes[actor_name].setChecked(1-actor.GetVisibility())
        actor.SetVisibility(1-actor.GetVisibility())

    def add_plane_widget_to_widget_menu(self, widget_name):
        menu_action = self.widget_menu.addAction(
            widget_name, self.plane_widget_toggle(widget_name))
        menu_action.setCheckable(True)
        menu_action.setChecked(self.interactor.widgets[widget_name].GetEnabled())

    def setup_isosurfaces(self):
        self.setup_boundaries()
        self.setup_defects()

    def setup_defects(self):
        self.colors["defects"] = self.settings["defects_color"]
        self.add_isosurface_slider(
            "order",
            actor_name="defects",
            mesh=None,
            label_txt="defects S",
            min_val=0,
            max_val=1,
            init_slider_int=30
        )



    def setup_boundaries(self):
        boundary_vis_kwargs = {
            "pbr":True,
            "metallic":1,
            "roughness":0.5,
            "diffuse":1,
            "smooth_shading":True
        }

        actor_name="boundaries (all)"
        self.colors[actor_name] = self.settings["boundaries_color"]
        self.refresh_functions[actor_name] = lambda: self.update_isosurface(
            actor_name, dataset_name="nematic_sites", contour_values=[0.5], **boundary_vis_kwargs,
        )
        self.refresh_functions[actor_name]()


        for i in range(1,self.num_boundaries+1):
            bdy = f"boundary_{i}"
            self.colors[bdy] = self.settings["boundaries_color"]
            self.refresh_functions[bdy] = lambda: self.update_isosurface(
                bdy, dataset_name=bdy, contour_values=[0.5], **boundary_vis_kwargs)
            self.refresh_functions[bdy]()
            self.isosurfaces[bdy]["actor"].SetVisibility(0)

    def setup_QSliders(self):
        self.QSliders_toolbar = qw.QToolBar('QSliders')
        self.QSliders_toolbar.setFixedWidth(150)
        self.app_window.addToolBar(Qt.Qt.LeftToolBarArea, self.QSliders_toolbar)

        self.toolbars["lighting"] = qw.QToolBar(
            'Lighting',
            orientation=Qt.Qt.Vertical,
            movable=True, floatable=True
        )
        toolbar = self.toolbars["lighting"]
        toolbar.setFixedWidth(self.QSliders_toolbar.width())
        self.QSliders["lighting"] = qw.QSlider(minimum=0, maximum=25,
                                          orientation=Qt.Qt.Horizontal)
        self.QSliders_labels["lighting"] = qw.QLabel()

        toolbar.addWidget(self.QSliders_labels["lighting"])
        slider = self.QSliders["lighting"]
        slider.valueChanged.connect(self.set_lights_intensity)
        slider.valueChanged.connect(
            lambda value: self.QSliders_labels["lighting"].setText(f'Lighting: {value}'))
        toolbar.addWidget(slider)
        slider.setValue(9)
        slider.setValue(8)
        self.app_window.addToolBar(Qt.Qt.LeftToolBarArea, toolbar)

        self.toolbars["director"] = qw.QToolBar(
            "Director", orientation=Qt.Qt.Vertical,
            movable=True, floatable=True
        )
        toolbar = self.toolbars["director"]
        toolbar.setFixedWidth(self.QSliders_toolbar.width())
        toolbar.addWidget(qw.QLabel('Director:'))
        self.QSliders_labels["director glyphs stride"] = qw.QLabel()
        toolbar.addWidget(self.QSliders_labels["director glyphs stride"])
        self.QSliders["director glyphs stride"] = qw.QSlider(
            minimum=1, maximum=int(max(1,max(np.array([self.Lx,self.Ly,self.Lz])/6))),
            orientation=Qt.Qt.Horizontal
        )
        slider = self.QSliders["director glyphs stride"]
        slider.setValue(self.director_resolution)
        slider.valueChanged.connect(self.make_coarsemesh)
        slider.valueChanged.connect(
            lambda value: self.QSliders_labels["director glyphs stride"].setText(
                f"  glyphs: stride={value}"
            )
        )
        self.wiggle_slider_to_update(slider)

        toolbar.addWidget(slider)
        self.add_viscolor_toolbar(
            'director',
            parent_toolbar=toolbar
        )

        toolbar.addWidget(qw.QLabel('  plane'))
        self.add_viscolor_toolbar(
            'slice_plane',
            parent_toolbar=toolbar
        )

        num_slider_divs = int(2*np.pi*1000)
        self.QSliders["director slice theta"] = qw.QSlider(
            minimum=0, maximum=num_slider_divs,
            orientation=Qt.Qt.Horizontal
        )
        slice_theta_slider_formula = lambda value: value * np.pi/num_slider_divs
        slice_theta_slider_inv_formula = lambda value: int(value / (np.pi/num_slider_divs))
        self.QSliders["director slice theta"] .valueChanged.connect(
            lambda value: self.alter_plane_widget(
                self.widgets['director_slice_widget'],
                self.director_slice_func,
                theta=slice_theta_slider_formula(value)
            )
        )

        self.QSliders_input_boxes["director slice theta"] = qw.QDoubleSpinBox(minimum=0, maximum=np.pi, decimals=3)
        self.QSliders_input_boxes["director slice theta"].editingFinished.connect(
            lambda: self.QSliders["director slice theta"] .setValue(
                slice_theta_slider_inv_formula(self.QSliders_input_boxes["director slice theta"].value())
            )
        )
        self.QSliders["director slice theta"] .valueChanged.connect(
            lambda value: self.QSliders_input_boxes["director slice theta"].setValue(slice_theta_slider_formula(value))
        )

        self.toolbars["director slice theta"] = qw.QToolBar('director slice theta')
        toolbar = self.toolbars["director slice theta"]
        toolbar.addWidget(qw.QLabel('θ'))
        toolbar.addWidget(self.QSliders_input_boxes["director slice theta"])
        toolbar.addWidget(self.QSliders["director slice theta"])


        self.QSliders["director slice phi"] = qw.QSlider(minimum=0, maximum=num_slider_divs, orientation=Qt.Qt.Horizontal)
        slice_phi_slider_formula = lambda value: value * 2*np.pi/num_slider_divs
        slice_phi_slider_inv_formula = lambda value: int(value / (2*np.pi/num_slider_divs))
        self.QSliders["director slice phi"].valueChanged.connect(lambda value: self.alter_plane_widget(
            self.widgets['director_slice_widget'],
            self.director_slice_func,
            phi=slice_phi_slider_formula(value)
        ))

        self.QSliders_input_boxes["director slice phi"] = qw.QDoubleSpinBox(minimum=0, maximum=2*np.pi, decimals=3)
        self.QSliders_input_boxes["director slice phi"].editingFinished.connect(
            lambda: self.QSliders["director slice phi"].setValue(
                slice_phi_slider_inv_formula(
                    self.QSliders_input_boxes["director slice phi"].value()
                )
            )
        )
        self.QSliders["director slice phi"].valueChanged.connect(
            lambda value: self.QSliders_input_boxes["director slice phi"].setValue(slice_phi_slider_formula(value))
        )

        self.toolbars["director slice phi"] = qw.QToolBar()
        toolbar = self.toolbars["director slice phi"]
        toolbar.addWidget(qw.QLabel('φ'))
        toolbar.addWidget(self.QSliders_input_boxes["director slice phi"])
        toolbar.addWidget(self.QSliders["director slice phi"])


        max_slice_translate = int(np.sqrt(np.sum((np.array(self.dims())/2)**2)))
        self.QSliders["director slice translate"] = qw.QSlider(
            minimum=-max_slice_translate,
            maximum=max_slice_translate,
            orientation=Qt.Qt.Horizontal
        )
        self.QSliders["director slice translate"].valueChanged.connect(
            lambda value: self.alter_plane_widget(
                self.widgets['director_slice_widget'],
                self.director_slice_func,
                origin=(
                    0.5*np.array(self.dims())
                    + value*np.array(
                        self.widgets['director_slice_widget'].GetNormal()
                    )
                )
            )
        )

        self.QSliders_input_boxes["director slice translate"] = qw.QSpinBox(
            minimum=-max_slice_translate, maximum=max_slice_translate
        )
        self.QSliders_input_boxes["director slice translate"].editingFinished.connect(
            lambda: self.QSliders["director slice translate"].setValue(
                self.QSliders_input_boxes["director slice translate"].value())
        )
        self.QSliders["director slice translate"].valueChanged.connect(
            lambda value: self.QSliders_input_boxes["director slice translate"].setValue(value)
        )
        self.toolbars["director slice translate"] = qw.QToolBar()
        toolbar = self.toolbars["director slice translate"]
        toolbar.addWidget(qw.QLabel('transl.'))
        toolbar.addWidget(self.QSliders_input_boxes["director slice translate"])
        toolbar.addWidget(self.QSliders["director slice translate"])

        self.toolbars["director slice"] = qw.QToolBar(orientation=Qt.Qt.Vertical)
        slice_toolbar = self.toolbars["director slice"]
        slice_toolbar.addWidget(self.toolbars["director slice theta"])
        slice_toolbar.addWidget(self.toolbars["director slice phi"])
        slice_toolbar.addWidget(self.toolbars["director slice translate"])

        slice_toolbar.setFixedWidth(self.QSliders_toolbar.width())
        self.toolbars["director"].addWidget(slice_toolbar)

        self.app_window.addToolBar(Qt.Qt.LeftToolBarArea, self.toolbars["director"])


    def setup_director_slice_widget(self):
        widget_name="director_slice_widget"
        if widget_name in self.widgets.keys():
            widget = self.widgets[widget_name]
            normal=widget.GetNormal()
            origin=widget.GetOrigin()
            enabled=widget.GetEnabled()
            widget.SetEnabled(0)
        else:
            normal=(1.0, 0.0, 0.0)
            origin=(self.Lx/2, self.Ly/2, self.Lz/2)
            enabled=True

        self.widgets[widget_name] = self.add_plane_widget(
            self.director_slice_func,
            factor=1.1,
            color=self.settings["plane_widget_color"],
            tubing=True,
            normal=normal,
            origin=origin
        )
        self.widgets[widget_name].SetEnabled(enabled)


    def Q33_from_Q5(self, Q5):
        (Qxx, Qxy, Qxz, Qyy, Qyz) = Q5.T
        Qmat = np.moveaxis(np.array([
                [Qxx, Qxy, Qxz],
                [Qxy, Qyy, Qyz],
                [Qxz, Qyz, -Qxx-Qyy]
                ]), -1, 0)
        return Qmat

    def n_from_Q(self, Qmat):
        """Get director from 3x3-matrix Q-tensor data"""
        evals, evecs = np.linalg.eigh(Qmat)
        return evecs[:,:,2]

    def set_settings(self, user_settings):
        self.settings = {
            "boundaries_color":"gray",
            "director_color":"red",
            "director_resolution":2,
            "default_defect_S":0.3, # initialization order value for defect isosurfaces
            "defects_color":(37/256,150/256,190/256),
            "checkbox_size":50, # size of toggle boxes in pixels
            "checkbox_spacing":10, # spacing between toggle boxes in pixels
            "window_size":(1200,800), # window size in pixels
            "cylinder_resolution":8, # angular resolution for each cylindrical rod; larger values look nicer but take longer to compute
            "slice_plane_color":"lightyellow", # set to None to use slice_color_function instead
            "slice_cmap":"cividis", # color map for use with slice_color_function
            "slice_color_function":(lambda slc: np.abs(slc["director"][:,0])), # optionally color slice plane by some function of director or order
            "plane_widget_color":"orange",
            "default_isosurface_color":"purple",
            "scalar_bar_args":dict(
                interactive=True,
                vertical=True,
                color="black",
                title_font_size=14,
                label_font_size=12,
                n_labels=3,
                # font_family="arial",
                height=50,
                n_colors=1000,
                # theme=self.theme,
                fmt="%.3f"
            ),
            "visible_symbol":'👁',
            "invisible_symbol":'🙈',
            "scalar_bar_maxheight":500,
            "scalar_bar_maxwidth":100,
            "scalar_bar_text_pad":5
        }
        self.colors["director"] = self.settings["director_color"]
        self.colors["slice_plane"] = self.settings["slice_plane_color"]
        for key, value in zip(user_settings.keys(), user_settings.values()):
            self.settings[key] = value
        self.director_resolution = self.settings["director_resolution"]

    def einstein_sum(self, sum_str, *arrays):
        sum_str = "..."+sum_str
        sum_str = sum_str.replace(" ", "")
        sum_str = sum_str.replace(",", ",...")
        sum_str = sum_str.replace("->", "->...")
        answer = np.einsum(sum_str, *arrays)
        answer = answer.reshape((np.prod(self.dims()),) + answer.shape[3:])
        return answer

    def meshdata_from_file(self, dat):

        # name the data columns:
        self.coords = dat[:,:3]
        self.Qdata = dat[:,3:8]
        self.Q33 = self.Q33_from_Q5(self.Qdata) # 3x3 Q matrix
        self.site_types = dat[:,8]
        self.order = dat[:,9]

        # grid for pyvista:
        self.fullmesh = pv.UniformGrid((self.Lx,self.Ly,self.Lz))
        # Order (defects) data:
        self.order[self.site_types>0] = np.max(self.order) # Don't plot defects inside objects
        self.fullmesh["order"] = self.order

        # director data:
        self.Qdata[self.site_types>0] = 0. # Don't bother calculating eigenvectors inside objects
        self.fullmesh["director"] = self.n_from_Q(self.Q33)

        # boundaries:
        self.fullmesh["nematic_sites"] = 1*(self.site_types<=0)
        self.num_boundaries = int(np.max(self.site_types))
        for i in range(1,self.num_boundaries+1):
            self.fullmesh[f"boundary_{i}"] = 1*(self.site_types==i)


        self.Qij = self.Q33.reshape(self.dims()+(3,3))
        self.diQjk = np.moveaxis(
            np.array(
                [np.roll(self.Qij,-1, axis=i)
                 - np.roll(self.Qij,1, axis=i)
                 for i in range(3)]
            ),
            0, -3
        )

        self.n = self.fullmesh["director"].reshape(self.dims() + (3,))
        self.dinj = np.empty(self.n.shape+(3,))
        for i in range(3):
            arr_up = np.roll(self.n, -1, axis=i)
            arr_dn = np.roll(self.n, 1, axis=i)
            sign_correction = np.sign(np.sum(arr_up*arr_dn, axis=-1))
            for j in range(3):
                arr_dn[...,j] *= sign_correction
            self.dinj[...,i] = arr_up - arr_dn
        self.dinj = np.swapaxes(self.dinj, -1,-2)

        # energy:
        self.fullmesh["energy_L1"] = (
            self.einstein_sum("ijk, ijk", self.diQjk, self.diQjk)
        )
        self.fullmesh["energy_L2"] = (
            self.einstein_sum("iik, jjk", self.diQjk, self.diQjk)
        )
        self.fullmesh["energy_L6"] = (
            self.einstein_sum("ij,ikl,jkl", self.Qij, self.diQjk, self.diQjk)
        )
        self.fullmesh["energy_L3"] = (
            self.einstein_sum("ijk, kij", self.diQjk, self.diQjk)
        )
        self.fullmesh["energy_L24"] = self.fullmesh["energy_L2"] - self.fullmesh["energy_L3"] # diQjk dkQij - diQij dkQjk
        for i in [1,2,3,6,24]:
            self.fullmesh[f"energy_L{i}"] *= 1*(self.site_types==0)
        L1 = self.fullmesh["energy_L1"]
        L2 = self.fullmesh["energy_L2"]
        L3 = self.fullmesh["energy_L3"]
        L6 = self.fullmesh["energy_L6"]
        S = self.fullmesh["order"]
        # TODO: account for q0
        # self.fullmesh["energy_K1"] = 2/(9*S*S) * (-L1/3 + 2*L2 -2/(3*S)*L6)
        self.fullmesh["splay"] = self.einstein_sum("ii", self.dinj)
        self.fullmesh["splay_vec"] = self.einstein_sum(
            "i, jj", self.n, self.dinj
        )
        self.fullmesh["energy_K1"] = self.fullmesh["splay"]**2
        # self.fullmesh["energy_K2"] = 2/(9*S*S) * (L1 - 2*L3)
        levi_civita = np.zeros((3,3,3), dtype=int)
        levi_civita[0,1,2] = levi_civita[1,2,0] = levi_civita[2,0,1] = 1
        levi_civita[0,2,1] = levi_civita[2,1,0] = levi_civita[1,0,2] = -1

        self.fullmesh["twist"] = self.einstein_sum("i,ijk,jk", self.n, levi_civita, self.dinj)

        self.fullmesh["energy_K2"] = (self.fullmesh["twist"] - self.q0)**2

        # self.fullmesh["energy_K3"] = 2/(9*S*S) * (L1/3 + 2/(3*S)*L6)
        self.fullmesh["bend"] = self.einstein_sum("i, ij", self.n, self.dinj)
        self.fullmesh["energy_K3"] = np.sum(self.fullmesh["bend"]**2, axis=-1)
        self.fullmeshes.append(self.fullmesh)
        for i, dataset_name in enumerate(["|n_x|", "|n_y|", "|n_z|"]):
            self.fullmesh[dataset_name] = np.abs(self.fullmesh["director"][...,i])
        self.fullmesh["active_force"] = (
            self.fullmesh["splay_vec"] - self.fullmesh["bend"]
        )
        self.fullmesh["energy_K24"] = (
            self.einstein_sum("ij,ji", self.dinj, self. dinj)
            - self.fullmesh["energy_K3"]
        )


    def make_coarsemesh(self, nres=None):
        if nres is None:
            nres = self.director_resolution
        else:
            self.director_resolution = nres
        self.coarsemesh = self.fullmesh.probe(
            pv.UniformGrid(tuple([ int(item/nres) for item in self.dims()]),
                           (nres,)*3
                          )
        )
        try:
            self.QSliders_labels["director glyphs stride"].setText(f"director stride: {nres}")
        except KeyError:
            pass

        self.director_slice_func = self.make_director_slice_func()

        if not 'director_slice_widget' in self.widgets.keys():
            self.setup_director_slice_widget()

        widget = self.widgets['director_slice_widget']
        self.refresh_functions['director'] = lambda : self.director_slice_func(
            widget.GetNormal(),
            widget.GetOrigin()
        )
        self.refresh_functions['slice_plane'] = self.refresh_functions['director']
        self.refresh_functions['director']()

    def plane_widget_toggle(self, widget_name):
        def return_function():
            if widget_name in self.widgets.keys():
                widget = self.widgets[widget_name]
                if widget.GetEnabled():
                    widget.EnabledOff()
                else:
                    widget.EnabledOn()
        return return_function


    def color_or_scalars(self, color):
        if color in self.scalar_fields():
            mesh_color_scalars = color
            color = None
        else:
            mesh_color_scalars = None
        return color, mesh_color_scalars

    def make_director_slice_func(self):
        def director_slice_func(normal, origin):
            """make glyph plot and transparent plane for director field slice"""
            origin = tuple(
                self.director_resolution * np.asarray(
                    np.array(origin)/self.director_resolution, dtype=int
                )
            )
            slc = self.coarsemesh.slice(normal=normal, origin=origin)
            cylinders = slc.glyph(orient="director", scale="nematic_sites",
                factor=self.director_resolution,
                geom=pv.Cylinder(
                    radius=0.2, height=1,
                    resolution=self.settings["cylinder_resolution"]
                ),
                tolerance=None
            )
            try:
                director_vis = self.renderer.actors["director"].GetVisibility()
                slice_plane_vis = self.renderer.actors["slice_plane"].GetVisibility()
            except KeyError:
                director_vis = 1
                slice_plane_vis = 1

            n_color, n_mesh_color_scalars = self.color_or_scalars(
                self.colors["director"]
            )

            try:
                self.renderer.actors["director"] = self.add_mesh(
                    cylinders,
                    color=n_color,
                    scalars=n_mesh_color_scalars,
                    scalar_bar_args=self.settings["scalar_bar_args"],
                    pbr=True, metallic=0.5, roughness=0.25, diffuse=1,
                    name="director" # "name" == actor's name so old actor is replaced
                )
            except ValueError:
                pass
            else:
                self.renderer.actors["director"].SetVisibility(director_vis)
                if "director" in self.visibility_checkboxes.keys():
                    checkbox = self.visibility_checkboxes["director"]
                    checkbox.toggled.disconnect()
                    checkbox.toggled.connect(
                        lambda: self.toggle_visibility("director")
                    )
                    checkbox.toggled.connect(self.set_checkbox_symbol(checkbox))

            slc_color, slc_mesh_color_scalars = self.color_or_scalars(
                self.colors["slice_plane"]
            )

            try:
                self.renderer.actors["slice_plane"] = self.add_mesh(
                    slc, opacity=0.01,
                    ambient=1, diffuse=0, specular=0, # glows, doesn't reflect
                    color=slc_color,
                    scalars=slc_mesh_color_scalars,
                    scalar_bar_args=self.settings["scalar_bar_args"],
                    cmap=self.settings["slice_cmap"],
                    name="slice_plane" # "name" == actor's name so old actor is replaced
                )
            except ValueError:
                pass
            else:
                self.renderer.actors["slice_plane"].SetVisibility(slice_plane_vis)
                if "slice_plane" in self.visibility_checkboxes.keys():
                    checkbox = self.visibility_checkboxes["slice_plane"]
                    checkbox.toggled.disconnect()
                    checkbox.toggled.connect(
                        lambda: self.toggle_visibility("slice_plane")
                    )
                    checkbox.toggled.connect(self.set_checkbox_symbol(checkbox))

            for mesh_color_scalars in [n_mesh_color_scalars, slc_mesh_color_scalars]:
                if mesh_color_scalars is not None:
                    self.standardize_scalar_bar(self.scalar_bars[mesh_color_scalars])

        return director_slice_func

    def set_color(self, actor_name, color):
        if not actor_name in self.renderer.actors:
            actor_name += "_isosurface"
        self.colors[actor_name] = color
        if actor_name in self.colorbars.keys():
            old_scalar_bar_name = self.colorbars[actor_name]
            self.remove_scalar_bar(old_scalar_bar_name)
        if actor_name in self.QSliders.keys():
            self.wiggle_slider_to_update(
                self.QSliders[actor_name]
            )
        elif actor_name in self.refresh_functions.keys():
            self.refresh_functions[actor_name]()

    def wiggle_slider_to_update(self, slider):
            val = slider.value()
            try:
                for i in [1,-1]:
                    slider.setValue(val+i)
                    slider.setValue(val)
            except:
                for i in [-1,1]:
                    slider.setValue(val+i)
                    slider.setValue(val)


    def set_checkbox_symbol(self, checkbox):
        def return_function():
            checkbox.setText(
                self.settings["visible_symbol"]
                if checkbox.checkState()
                else self.settings["invisible_symbol"]
            )
        return return_function

    def update_isosurface(self, actor_name, dataset_name = None, contour_values=None, **contour_kwargs):


        if not actor_name in self.isosurfaces.keys():
            self.isosurfaces[actor_name] = dict(
                contour_kwargs = dict(
                    name=actor_name
                )
            )
        isosurface_dict = self.isosurfaces[actor_name]


        if dataset_name is None:
            if not "dataset_name" in isosurface_dict.keys():
                print("Error: no dataset_name passed to update_isosurface")
                raise AttributeError
        else:
            isosurface_dict["dataset_name"] = dataset_name

        if contour_values is None:
            if not "contour_values" in isosurface_dict.keys():
                isosurface_dict["contour_values"] = [0.5]
            # else keep existing contour values
        else:
            if type(contour_values) is not list:
                contour_values = [contour_values]
            isosurface_dict["contour_values"] = contour_values

        # if "actor_name" in isosurface_dict.keys():
        #     actor_name = isosurface_dict["actor_name"]
        # else:
        #     actor_name = isosurface_dict["dataset_name"] + "_isosurface"

        if "color" in contour_kwargs.keys(): # for newly passed color setting
            color = contour_kwargs["color"]
        elif "color" in isosurface_dict["contour_kwargs"].keys():
            # for previously set color
            color = isosurface_dict["contour_kwargs"]["color"]
            if color is None and "scalars" in isosurface_dict["contour_kwargs"].keys():
                color = isosurface_dict["contour_kwargs"]["scalars"]
        else:
            color = None
        if color is None:
            if actor_name in self.colors.keys():
                color = self.colors[actor_name]
            else:
                color = self.settings["default_isosurface_color"]
        self.colors[actor_name] = color

        if color in self.scalar_fields(): # "color" is really "scalars"
            contour_kwargs["scalars"] = color
            # if not "show_scalar_bar" in contour_kwargs.keys():
            contour_kwargs["show_scalar_bar"] = True
            if not "scalar_bar_args" in contour_kwargs.keys():
                contour_kwargs["scalar_bar_args"] = self.settings["scalar_bar_args"]

            new_scalar_bar_title = actor_name + ": \n" + color
            if "title" in contour_kwargs["scalar_bar_args"].keys():
                # check for customized scalar bar title, keep unchanged if so
                if (actor_name + ": \n") in contour_kwargs["scalar_bar_args"]["title"]:
                    contour_kwargs["scalar_bar_args"]["title"] = new_scalar_bar_title
            else:
                # use standard scalar bar title if no title assigned yet
                contour_kwargs["scalar_bar_args"]["title"] = new_scalar_bar_title
            contour_kwargs["color"] = None
        else: # "color" is actually a color
            contour_kwargs["color"] = color
            contour_kwargs["scalars"] = None
            contour_kwargs["show_scalar_bar"] = False

        for key in contour_kwargs.keys():
            self.isosurfaces[actor_name]["contour_kwargs"][key] = contour_kwargs[key]


        # if color is None:
        #     # check if we've already assigned a color to this isosurface
        #     if actor_name in self.colors.keys():
        #         color = self.colors[actor_name]
        #     else:
        #         color = self.settings["default_isosurface_color"]
        #         self.colors[actor_name] = color
        #     contour_kwargs["color"] = color
        # elif color in self.scalar_fields():
        #     contour_kwargs["scalars"] = color
        #     contour_kwargs["show_scalar_bar"] = True
        #     contour_kwargs["scalar_bar_args"] = self.settings["scalar_bar_args"]
        # else:
        #     contour_kwargs["color"] = color
        #     contour_kwargs["scalars"] = None
        #     contour_kwargs["show_scalar_bar"] = False

        if "scalar_bar" in isosurface_dict.keys():
            scalar_bar_title = isosurface_dict["scalar_bar"].GetTitle()
            if scalar_bar_title in self.scalar_bars.keys():
                self.remove_scalar_bar(scalar_bar_title)
        self.renderer.actors[actor_name] = self.add_mesh(
            self.fullmesh.contour(
                isosurface_dict["contour_values"],
                scalars=isosurface_dict["dataset_name"]
            ),
            **self.isosurfaces[actor_name]["contour_kwargs"]
        )
        isosurface_dict["actor"] = self.renderer.actors[actor_name]
            # dataset_name = dataset_name,
            # contour_values = contour_value,
            # color=color

        if isosurface_dict["contour_kwargs"]["show_scalar_bar"]:
            scalar_bar = self.scalar_bars[
                isosurface_dict["contour_kwargs"]["scalar_bar_args"]["title"]
            ]
            self.standardize_scalar_bar(scalar_bar)
            isosurface_dict["scalar_bar"] = scalar_bar

    def generic_slider_callback(self, actor_name, scalars, contour_value,
        color=None, mesh_color_scalars=None, clim=None, cmap=None):
        """generic callback function for slider controlling isosurfaces"""
        kwargs={
            "show_scalar_bar":False,
            "smooth_shading":True,
            "name":actor_name,
            "pbr":True,
            "metallic":0.5,
            "roughness":0.25,
            "diffuse":1,
        }
        contour_value = max(np.min(self.fullmesh[scalars]), contour_value)
        contour_value = min(np.max(self.fullmesh[scalars]), contour_value)
        # kwargs["contour_values"] = [contour_value]
        # kwargs["actor_name"] = actor_name
        # kwargs["dataset_name"] = scalars
        # if color is None and mesh_color_scalars is None:
        #     if actor_name in self.colors.keys():
        #         color = self.colors[actor_name]
        #     else:
        #         color = self.settings["default_isosurface_color"]
        #         self.colors[actor_name] = color
        if mesh_color_scalars is not None:
            kwargs["color"] = mesh_color_scalars
        else:
            kwargs["color"] = color
        try:
            self.update_isosurface(actor_name, dataset_name=scalars, contour_values=[contour_value], **kwargs)
        #
        # if color is None and mesh_color_scalars is None:
        #     if actor_name in self.colors.keys():
        #         color = self.colors[actor_name]
        #     else:
        #         color = self.settings["default_isosurface_color"]
        #         self.colors[actor_name] = color
        # # check if "color" was mean to be "mesh_color_scalars"
        # if color in self.scalar_fields():
        #     mesh_color_scalars = color
        #     color = None
        #
        # if mesh_color_scalars is not None:
        #     kwargs["scalars"] = mesh_color_scalars
        #     kwargs["show_scalar_bar"] = True
        #     kwargs["scalar_bar_args"] = self.settings["scalar_bar_args"]
        #     kwargs["clim"] = clim
        #     if cmap is not None:
        #         kwargs["cmap"] = cmap
        #     color = None
        # kwargs["color"] = color
        # try:
        #     self.renderer.actors[actor_name] = self.add_mesh(
        #         self.fullmesh.contour(
        #             [contour_value], scalars=scalars
        #         ), **kwargs
        #     )
        #     actor = self.renderer.actors[actor_name]
        #     if actor_name in self.visibility_checkboxes.keys():
        #         checkbox = self.visibility_checkboxes[actor_name]
        #         checkbox.toggled.disconnect()
        #         checkbox.toggled.connect(
        #             lambda : actor.SetVisibility(1 - actor.GetVisibility())
        #         )
        #         checkbox.toggled.connect(self.set_checkbox_symbol(checkbox))
        #
        #     if kwargs["show_scalar_bar"]:
        #         scalar_bar_name = kwargs["scalars"]
        #         scalar_bar = self.scalar_bars[scalar_bar_name]
        #         self.standardize_scalar_bar(scalar_bar)
        #         self.colorbars[actor_name] = scalar_bar_name
        except ValueError: # ignore "empty mesh" warnings
            """
            When all values are the same, such as typically for "order" in the first frame,
            we can't try to draw isosurfaces. This is a silly hack to avoid problems
            with other functions expecting this isosurface actor to exist, by assigning
            it to point to the bounding box actor after deleting the isosurface we may
            have drawn previously for a different frame.
            """
            self.renderer.remove_actor(actor_name)
            self.renderer.actors[actor_name] = self.renderer.actors['outline']

    def add_isosurface_slider(self, dataset_name, actor_name=None,
                              mesh=None, color=None, label_txt=None,
                              min_val=None, max_val=None, init_slider_int = 50):
        if actor_name is None:
            actor_name = dataset_name+"_isosurface"
        while actor_name in self.renderer.actors.keys():
            actor_name += "\'"

        if mesh is None:
            mesh = self.fullmesh
        if label_txt is None:
            label_txt = dataset_name
        dataset=mesh[dataset_name]
        if min_val is None:
            min_val = np.min(dataset)
        if max_val is None:
            max_val = np.max(dataset)

#         self.QSliders_toolbar.addSeparator()
        self.toolbars[actor_name] = qw.QToolBar(
            actor_name,
            movable=True, floatable=True
        )
        self.QSliders_labels[actor_name], self.QSliders_updaters[actor_name], self.QSliders[actor_name] = self.add_QSlider(
            lambda value: self.generic_slider_callback(
               actor_name, dataset_name, value, color
            ),
            self.toolbars[actor_name],
            actor_name,
            scalars=mesh[dataset_name],
            label_txt=label_txt,
            min_val = min_val,
            max_val = max_val
        )
        try:
            self.QSliders[actor_name].setValue(init_slider_int+1)
        except:
            self.QSliders[actor_name].setValue(init_slider_int-1)
        self.QSliders[actor_name].setValue(init_slider_int)

        self.renderer.actors[actor_name].SetVisibility(1)
        self.add_viscolor_toolbar(actor_name, parent_toolbar=self.toolbars[actor_name])
        self.toolbars[actor_name].setFixedWidth(self.QSliders_toolbar.width())
        self.app_window.addToolBar(
            Qt.Qt.LeftToolBarArea,
            self.toolbars[actor_name]
        )
        # self.isosurfaces[actor_name] = self.renderer.actors[actor_name]



    def add_QSlider(self, update_method, toolbar, actor_name, num_divs=100,
                    init_val=50, min_val = None, max_val = None, scalars=None, label_txt=None):
        def slider_formula(slider_value):
            return min_val + (max_val-min_val)*slider_value/100
        def external_update(float_value):
            slider_value = int( 100*(float_value-min_val)/(max_val-min_val))
            slider.setValue(slider_value)

        slider = qw.QSlider(Qt.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(num_divs)
        slider.setValue(init_val)

        text_row = qw.QToolBar()
        if label_txt is not None:
            label = qw.QLabel(label_txt)
            text_row.addWidget(label)

        spinbox = qw.QDoubleSpinBox(
            minimum=slider_formula(slider.minimum()),
            maximum=slider_formula(slider.maximum()),
            value=slider_formula(init_val),
            singleStep=0.1,
            decimals=3
        )
        # spinbox.setDecimals(3)
        # spinbox.setSingleStep(0.1)
        # spinbox.setValue(slider_formula(init_val))
        def spinbox_callback():
            external_update(spinbox.value())
        spinbox.editingFinished.connect(spinbox_callback)
        text_row.addWidget(spinbox)
        toolbar.addWidget(text_row)


        if max_val is None and scalars is not None:
            max_val = np.max(scalars)
        if min_val is None and scalars is not None:
            min_val = np.min(scalars)

        def valuechange_method(slider_value):
            float_value = slider_formula(slider_value)
            if actor_name in self.renderer.actors:
                vis = self.renderer.actors[actor_name].GetVisibility()
            else:
                vis = 0
            update_method(float_value)
            self.renderer.actors[actor_name].SetVisibility(vis)
            spinbox.setValue(float_value)
        slider.valueChanged.connect(valuechange_method)
        toolbar.addWidget(slider)
        self.refresh_functions[actor_name] = lambda : self.wiggle_slider_to_update(slider)
        return label, external_update, slider

    def generic_menu_toggle(self, actor_name):
        def return_function():
            actor=self.renderer.actors[actor_name]
            self.toggle_visibility(actor_name)
            # actor.SetVisibility(1-actor.GetVisibility())
            if actor_name in self.QSliders.keys():
                self.wiggle_slider_to_update(self.QSliders[actor_name])
            if actor_name in self.scalar_bars:
                self.scalar_bars[actor_name].SetVisibility(actor.GetVisibility())
        return return_function

    def set_lights_intensity(self, intensity):
        for light in self.renderer.lights:
            light.SetIntensity(intensity)
#         self.lighting_slider_label.setText(f"lighting: {intensity}")

    def rescale_lights_intensity(self, factor):
        for light in self.renderer.lights:
            light.SetIntensity(factor*light.GetIntensity())

    def add_isosurface_slider_aux(self, scalars_name):
        def return_function():
            self.add_isosurface_slider(scalars_name)
        return return_function


    def add_slice_aux(self, scalars_name):
        def return_function():
            self.add_slice(scalars_name)
        return return_function

    def add_slice(self, scalars_name, slice_name=None, widget_name=None):
        if slice_name is None:
            slice_name = scalars_name+"_slice"
            while slice_name in self.elements.keys():
                slice_name += "\'"
        if widget_name is None:
            widget_name = scalars_name + "_widget"
            while widget_name in self.widgets.keys():
                widget_name += "\'"

        def refresh_callback():
            scalars = self.fullmesh[scalars_name]
            stdev = np.std(scalars)

            self.slices[slice_name] = self.add_mesh_slice(
                self.fullmesh,
                scalars=scalars_name, name=slice_name,
                ambient=1, specular=0, diffuse=0,
                cmap='jet',
                # clim=clim,
                #(lambda arr:
                #       [ np.average(arr) - ((-1)**i)*2*np.std(arr) for i in range(2) ]
                # )(self.fullmesh[scalars_name]),
                scalar_bar_args=self.settings["scalar_bar_args"]
            )
        self.refresh_functions[slice_name] = refresh_callback
        self.refresh_functions[slice_name]()
        self.standardize_scalar_bar(self.scalar_bars[scalars_name])
        self.elements[slice_name].SetVisibility(True)
        self.add_to_toggle_menu(slice_name)
        self.widgets[widget_name] = self.plane_widgets[-1]
        self.add_plane_widget_to_widget_menu(widget_name)

    def standardize_scalar_bar(self, scalar_bar):
        scalar_bar.SetMaximumHeightInPixels(
            self.settings["scalar_bar_maxheight"]
        )
        scalar_bar.SetMaximumWidthInPixels(
            self.settings["scalar_bar_maxwidth"]
        )
        scalar_bar.SetTextPad(self.settings["scalar_bar_text_pad"])
        scalar_bar.UnconstrainedFontSizeOff()

    def add_viscolor_toolbar(self, actor_name, label=None, parent_toolbar=None):
        toolbar_row = qw.QToolBar()
        if parent_toolbar is None:
            parent_toolbar = self.QSliders_toolbar
        if label is not None:
            parent_toolbar.addWidget(qw.QLabel(label))
        self.add_color_picker_button(actor_name, toolbar_row)
        self.add_visibility_checkbox(actor_name, toolbar_row)
        parent_toolbar.addWidget(toolbar_row)

    def color_picker(self, actor_name):
        self.colors[actor_name] = qw.QColorDialog.getColor().name()
        self.refresh_functions[actor_name]()

    def add_color_picker_button(self, actor_name, toolbar):
        color_button = qw.QPushButton('🎨')
        color_button.setToolTip('choose color for '+actor_name)
        color_button.clicked.connect(lambda: self.color_picker(actor_name))
        toolbar.addWidget(color_button)

    def add_visibility_checkbox(self, actor_name, toolbar):
        # on_symbol = '👁'
        # off_symbol = '🙈'
        checkbox = qw.QCheckBox(self.settings["visible_symbol"])
        checkbox.setChecked(self.renderer.actors[actor_name].GetVisibility())
        checkbox.toggled.connect(self.set_checkbox_symbol(checkbox))
        #     le: checkbox.setText(
        #         on_symbol
        #         if checkbox.checkState()
        #         else off_symbol
        #     )
        # )
        checkbox.toggled.connect(lambda : self.toggle_visibility(actor_name))
        checkbox.setToolTip('toggle visibility of '+actor_name)
        toolbar.addWidget(checkbox)
        self.visibility_checkboxes[actor_name] = checkbox


    def alter_plane_widget(self, widget, actor_update_func, theta=None, phi=None, origin=None, dtheta=0, dphi=0, dOrigin=(0,0,0)):
        Nx,Ny,Nz = widget.GetNormal()
        slice_origin = widget.GetOrigin()
        Ntheta = np.arccos(Nz)
        Nphi = np.arctan2(Ny, Nx)
        if theta is not None:
            Ntheta = theta
        if phi is not None:
            Nphi = phi
        if origin is not None:
            slice_origin = origin
        Ntheta += dtheta
        Nphi += dphi
        slice_origin = np.array(slice_origin) + np.array(dOrigin)
        widget.SetNormal(
            np.sin(Ntheta)*np.cos(Nphi),
            np.sin(Ntheta)*np.sin(Nphi),
            np.cos(Ntheta)
        )
        widget.SetOrigin(slice_origin)
        actor_update_func(widget.GetNormal(), widget.GetOrigin())


    def shift_plane_widget_along_normal(self, widget, actor_update_func, shift):
        self.alter_plane_widget(
            widget, actor_update_func,
            origin=0.5*np.array(
                self.dims()) + shift*np.array(widget.GetNormal()
            )
        )

    def fast_import(self,filename, sep="\t"):
        return np.array(read_csv(filename, sep=sep, header=None))

    def legacy_Qmin_import(self, Qtensor_filenames):
        if type(Qtensor_filenames) is str:
            Qtensor_filenames = [Qtensor_filenames]
        Qtensor_filenames = [
            filename for filename in Qtensor_filenames
            if "Qtensor" in filename
        ]
        for qfi, Qtensor_filename in enumerate(Qtensor_filenames):
            print(f"Beginning data import. If the program hangs here, try quitting the Python/Vista viewer application.", end="\r")
            print(f"Loading {Qtensor_filename} ({qfi+1} of {len(Qtensor_filenames)})                                   ", end="\r")
            qtensor = np.asarray(
                self.fast_import(
                    Qtensor_filename,
                    sep=" "),
                dtype=float
            )
            qmatrix = np.asarray(
                self.fast_import(
                    Qtensor_filename.replace('Qtensor', 'Qmatrix'),
                    sep=" "),
                dtype=float
            )

            # calculate nematic order here, not using file, because that info isn't
            # saved for animations
            S = np.linalg.eigvalsh(self.Q33_from_Q5(qmatrix))[:,-1]
            site_types = qtensor[:,-1]
            # infer Lx, Ly, Lz from filename
            self.Lx, self.Ly, self.Lz = np.asarray(
                (' '.join(Qtensor_filename.split('/')[-1].split('.dat')[0].split('x'))).split('_')[1].split(),
                dtype=int
            )
            data_num_lines = len(qtensor)
            inferred_data_stride = int((self.Lx*self.Ly*self.Lz/data_num_lines)**(1/3))
            if inferred_data_stride > 1:
                self.Lx = int(self.Lx/inferred_data_stride)
                self.Ly = int(self.Ly/inferred_data_stride)
                self.Lz = int(self.Lz/inferred_data_stride)
                #!!! replace this with incorporating stride into calculations
            Z, Y, X = np.meshgrid(range(self.Lz),range(self.Ly),range(self.Lx))
            open_Qmin_style_data = np.concatenate((
                np.array([X.flatten(), Y.flatten(), Z.flatten()]).T,
                qmatrix,
                np.array((site_types, S)).T
            ), axis=-1)
            self.data.append(open_Qmin_style_data)
            self.meshdata_from_file(open_Qmin_style_data) # sets self.fullmesh and appends it to self.fullmeshes
        if len(Qtensor_filenames) > 0 :
            self.init_procedures_after_data_import()

    def open_files_dialog(self, sort=True):
        dlg = qw.QFileDialog()
        dlg.setFileMode(qw.QFileDialog.ExistingFiles)
#         dlg.setFilter("Text files (*.txt *.dat)")
        filenames = dlg.getOpenFileNames()[0]
        if sort:
            filenames = sort_filenames_by_timestamp(filenames)
            # timestamps = [
            #     int(filename.split("_")[-1].split(".")[0])
            #     for filename in filenames
            # ]
            # for timestamp in sorted(timestamps):
            #     sorted_filenames.append(filenames[timestamps.index(timestamp)])
            # filenames = sorted_filenames
        if len(filenames) > 0:
            self.load(filenames)

    def show_params(self, the_dict, exclude_keys=["theme"]):
        return DataFrame(
            [
                [key, str(value)]
                for (key, value) in zip(the_dict.keys(), the_dict.values())
                if not key in exclude_keys
            ],
            columns=('key', 'value')
        )


def sort_filenames_by_timestamp(filenames):
    timestamps = []
    for filename in filenames:
        filename_trimmed = ''.join(filename.split(".")[:-1])
        filename_suffix = filename_trimmed.split("_")[-1]
        # check for open-Qmin mpi x#y#z# string, to ignore for timestamp
        if (
            "x" in filename_suffix
            and "y" in filename_suffix
            and "z" in filename_suffix
        ):
            check_for_mpi_suffix = (
                filename_suffix.replace("x","").replace("y","").replace("z","")
            )
            if check_for_mpi_suffix.isdigit():
                filename_suffix = filename_trimmed.split("_")[-2]
        if filename_suffix.isnumeric():
            timestamps.append(int(filename_suffix))
        else:
            # give up on sorting by timestamp
            return filenames

    sorted_filenames = []
    for timestamp in sorted(timestamps):
        sorted_filenames.append(filenames[timestamps.index(timestamp)])
    return sorted_filenames




if __name__ == '__main__':
    filenames = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            filenames += glob.glob(arg)
    if len(filenames) > 0:
        filenames = sort_filenames_by_timestamp(filenames)
        print('Found these files (and importing in this order):')
        for filename in filenames:
            print(filename)
        my_viewmin_plot = ViewMinPlot(filenames)
    else:
        my_viewmin_plot = ViewMinPlot()
    input("\nPress Enter in this window to quit...\n")
