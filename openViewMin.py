#!/usr/bin/env python3.9  ## <-- or replace with your path to Python<3.10

# TODO:
#
# * add animation
# * add multiple director glyph slices
# * add omega vector
# * check animation for open-Qmin data
# * add auto-stitching for open-Qmin mpi data
# * add data stride handling (including scaling derivatives)
# * add ellipses as optional replacement for cylinders
# * add "remove actor" option?
# * GUI option to add glyphs to a filter
# * GUI option to add

# Known issues:
# * scalar bar text looks gross... is its antialiasing disabled?

import numpy as np
from pandas import read_csv, DataFrame
from qtpy import QtWidgets as qw
import qtpy.QtCore as Qt
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import sys
import glob
import matplotlib.pyplot as plt

class ViewMinPlot(BackgroundPlotter):
    def __init__(self, filenames=[], user_settings={}):
        super().__init__(multi_samples=8,line_smoothing=True, point_smoothing=True, polygon_smoothing=True,
        )
        self.theme.antialiasing = True
        self.finished_setup = False
        self.make_empty_convenience_arrays()
        self.make_geometric_objects_dict()
        self.set_settings(user_settings)
        self.app_window.setWindowTitle("openViewMin nematic visualization environment")
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
        self.QSliders = {}
        self.QSliders_labels = {}
        self.QSliders_updaters = {}
        self.QSliders_input_boxes = {}
        self.toolbars = {}
        self.colors = {}
        self.refresh_functions = {}
        self.colorbars = {}
        self.actors_dict = {}
        self.viscolor_toolbars = {}

    def make_geometric_objects_dict(self):
        self.geometric_objects = dict(
            cylinder = pv.Cylinder,
            arrow = pv.Arrow,
            sphere = pv.Sphere,
            plane = pv.Plane,
            line = pv.Line,
            box = pv.Box,
            cone = pv.Cone,
            polygon = pv.Polygon,
            disc = pv.Disc
        )

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
        btn_width=35
        self.animation_buttons_toolbar = qw.QToolBar(
            "Animation",
            orientation=Qt.Qt.Vertical
        )
        tb = self.animation_buttons_toolbar
        tb.setStyleSheet("QToolBar{spacing:0px;}")

        toolbar_row = qw.QToolBar(orientation=Qt.Qt.Horizontal)
        toolbar_row.setStyleSheet("QToolBar{spacing:0px;}")
        sb = qw.QSpinBox(minimum=0, maximum=len(self.fullmeshes))
        sb.setValue(self.frame_num)
        sb.editingFinished.connect(lambda : self.load_frame(frame_num=sb.value()))
        self.frame_spinbox = sb

        btn = qw.QPushButton('|<')
        btn.setFixedWidth(btn_width)
        btn.released.connect(self.first_frame)
        btn.released.connect(lambda : sb.setValue(self.frame_num))
        toolbar_row.addWidget(btn)

        btn = qw.QPushButton('<')
        btn.setFixedWidth(btn_width)
        btn.released.connect(self.previous_frame)
        btn.released.connect(lambda : sb.setValue(self.frame_num))
        toolbar_row.addWidget(btn)

        # toolbar_row.addWidget(sb)

        btn = qw.QPushButton('>')
        btn.released.connect(self.next_frame)
        btn.released.connect(lambda : sb.setValue(self.frame_num))
        btn.setFixedWidth(btn_width)
        toolbar_row.addWidget(btn)

        btn = qw.QPushButton('>|')
        btn.released.connect(self.last_frame)
        btn.released.connect(lambda : sb.setValue(self.frame_num))
        btn.setFixedWidth(btn_width)
        toolbar_row.addWidget(btn)

        tb.addWidget(toolbar_row)
        tb.setFixedWidth(self.settings["QSliders_toolbar_width"])
        self.app_window.addToolBar(Qt.Qt.LeftToolBarArea, tb)
        # self.app_window.addToolBar(tb)


    def setup_import_menu(self):
        self.the_import_menu = self.main_menu.addMenu('Import')
        self.the_import_menu.addAction(
            'Open file(s)...',
            self.open_files_dialog
        )

    def setup_menus(self):
        # self.toggle_menu = self.main_menu.addMenu('Toggle')
        # self.toggle_menu.aboutToShow.connect(self.update_toggle_menu)
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
        # self.the_color_by_menu = self.main_menu.addMenu("Color by")
        # self.the_color_by_menu.aboutToShow.connect(self.color_by_menu_update)
        # for scalar_bar_name in self.scalar_bars.keys():
        #     self.add_scalar_bar_to_toggle_menu(scalar_bar_name)

    # def color_by_menu_update(self):
    #     menu = self.the_color_by_menu
    #     menu.clear()
    #     for actor_name in self.colors.keys():
    #         # exclude color bars and widget outlines
    #         if not "Addr=" in actor_name and not (
    #                 len(actor_name.split('outline'))==2
    #                 and len(actor_name.split('outline')[0])>0
    #         ):
    #             submenu = menu.addMenu(actor_name)
    #             submenu.aboutToShow.connect(self.color_by_submenu_update(actor_name, submenu))


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

    # def update_toggle_menu(self):
    #     self.toggle_menu.clear()
    #     for actor_name in self.renderer.actors.keys():
    #         # exclude color bars and widget outlines
    #         if (not "Addr=" in actor_name
    #             and not (len(actor_name.split('outline'))==2
    #                      and len(actor_name.split('outline')[0])>0
    #                     )
    #         ):
    #             self.add_to_toggle_menu(actor_name)
    #     for scalar_bar_name in self.scalar_bars.keys():
    #         self.add_scalar_bar_to_toggle_menu(scalar_bar_name)


    def update_widget_menu(self):
        self.widget_menu.clear()
        for widget_name in self.widgets.keys():
            self.add_plane_widget_to_widget_menu(widget_name)

    # def add_to_toggle_menu(self, actor_name):
    #     menu_action = self.toggle_menu.addAction(actor_name,
    #                           self.generic_menu_toggle(actor_name))
    #     menu_action.setCheckable(True)
    #     is_visible = self.renderer.actors[actor_name].GetVisibility()
    #     menu_action.setChecked(is_visible)

    # def add_scalar_bar_to_toggle_menu(self, scalar_bar_name):
    #     # scalar_bar = self.scalar_bars[scalar_bar_name]
    #     menu_action = self.toggle_menu.addAction("└─ " + scalar_bar_name + " colorbar",
    #         lambda : self.toggle_visibility(scalar_bar_name, is_scalar_bar=True)
    #     )
    #     menu_action.setCheckable(True)
    #     is_visible = self.scalar_bars[scalar_bar_name].GetVisibility()
    #     menu_action.setChecked(is_visible)


    def toggle_visibility(self, actor_name, is_scalar_bar=False):
        # if type(actor) is str:
        #     actor_name = actor
        if is_scalar_bar:
            actor = self.scalar_bars[actor_name]
        else:
            actor = self.renderer.actors[actor_name]
        if actor_name in self.visibility_checkboxes.keys():
            self.visibility_checkboxes[actor_name].setChecked(1-actor.GetVisibility())
        actor.SetVisibility(1-actor.GetVisibility())
        if actor_name in self.colorbars:
            scalar_bar = self.scalar_bars[self.colorbars[actor_name]]
            if scalar_bar.GetVisibility() and not actor.GetVisibility():
                scalar_bar.SetVisibility(0)

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
            label_txt="Defects: S=",
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

        self.toolbars["boundaries"] = qw.QToolBar("Boundaries",
            orientation=Qt.Qt.Vertical,
            movable=True, floatable=True
        )
        toolbar = self.toolbars["boundaries"]
        toolbar.setFixedWidth(self.settings["QSliders_toolbar_width"])
        toolbar.addWidget(qw.QLabel("Boundaries (all)"))
        self.add_viscolor_toolbar(
            "boundaries (all)", parent_toolbar=toolbar
        )
        self.app_window.addToolBar(Qt.Qt.LeftToolBarArea, toolbar)

        for i in range(1,self.num_boundaries+1):
            bdy = f"boundary_{i}"
            self.colors[bdy] = self.settings["boundaries_color"]
            self.refresh_functions[bdy] = lambda: self.update_isosurface(
                bdy, dataset_name=bdy, contour_values=[0.5], **boundary_vis_kwargs)
            self.refresh_functions[bdy]()
            self.actors_dict[bdy]["actor"].SetVisibility(0)

    def setup_QSliders(self):
        # self.QSliders_toolbar = qw.QToolBar('QSliders')
        # self.QSliders_toolbar.setFixedWidth(150)
        # self.app_window.addToolBar(Qt.Qt.LeftToolBarArea, self.QSliders_toolbar)

        self.toolbars["lighting"] = qw.QToolBar(
            'Lighting',
            orientation=Qt.Qt.Vertical,
            movable=True, floatable=True
        )
        toolbar = self.toolbars["lighting"]
        toolbar.setFixedWidth(self.settings["QSliders_toolbar_width"])
        toolbar_row = qw.QToolBar(orientation=Qt.Qt.Horizontal)
        self.QSliders["lighting"] = qw.QSlider(minimum=0, maximum=20,
                                          orientation=Qt.Qt.Horizontal)
        self.QSliders_labels["lighting"] = qw.QLabel()

        # toolbar.addWidget(self.QSliders_labels["lighting"])
        slider = self.QSliders["lighting"]
        slider.valueChanged.connect(self.set_lights_intensity)
        slider.valueChanged.connect(
            lambda value: self.QSliders_labels["lighting"].setText(f'💡: {value}'))
        slider.setFixedWidth(80)
        toolbar_row.addWidget(self.QSliders_labels["lighting"])
        toolbar_row.addWidget(slider)
        toolbar.addWidget(toolbar_row)

        # toolbar.addWidget(slider)
        slider.setValue(9)
        slider.setValue(8)
        self.app_window.addToolBar(Qt.Qt.LeftToolBarArea, toolbar)

        self.toolbars["boundaries"] = qw.QToolBar(
            "Boundaries", orientation=Qt.Qt.Vertical,
            movable=True, floatable=True
        )

        self.toolbars["director"] = qw.QToolBar(
            "Director", orientation=Qt.Qt.Vertical,
            movable=True, floatable=True
        )
        toolbar = self.toolbars["director"]
        toolbar.setFixedWidth(self.settings["QSliders_toolbar_width"])
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

        toolbar.setFixedWidth(self.settings["QSliders_toolbar_width"])
        self.toolbars["director"].addWidget(slice_toolbar)

        self.app_window.addToolBar(Qt.Qt.LeftToolBarArea, self.toolbars["director"])


    def setup_director_slice_widget(self):
        self.setup_slice_widget(
            "director_slice_widget",
            self.director_slice_func
        )
        # widget_name="director_slice_widget"
        # if widget_name in self.widgets.keys():
        #     widget = self.widgets[widget_name]
        #     normal=widget.GetNormal()
        #     origin=widget.GetOrigin()
        #     enabled=widget.GetEnabled()
        #     widget.SetEnabled(0)
        # else:
        #     normal=(1.0, 0.0, 0.0)
        #     origin=(self.Lx/2, self.Ly/2, self.Lz/2)
        #     enabled=True
        #
        # self.widgets[widget_name] = self.add_plane_widget(
        #     self.director_slice_func,
        #     factor=1.1,
        #     color=self.settings["plane_widget_color"],
        #     tubing=True,
        #     normal=normal,
        #     origin=origin
        # )
        # self.widgets[widget_name].SetEnabled(enabled)


    def setup_slice_widget(self, widget_name, callback):
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
            callback,
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
                height=50,
                n_colors=1000,
                fmt="%.3f"
            ),
            "visible_symbol":'👁',
            "invisible_symbol":'🙈',
            "scalar_bar_maxheight":500,
            "scalar_bar_maxwidth":100,
            "scalar_bar_text_pad":5,
            "default_mesh_kwargs":dict(
                pbr=True,
                metallic=0.5,
                roughness=0.25,
                diffuse=1
            ),
            "default_slice_kwargs":dict(
                opacity=0.9,
                ambient=1, diffuse=0, specular=0, # glows, doesn't reflect
            ),
            "QSliders_toolbar_width":150,
            "rod_aspect_ratio":5
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
        # try:
        #     self.QSliders_labels["director glyphs stride"].setText(f"director stride: {nres}")
        # except KeyError:
        #     pass

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

    def add_glyphs_to_mesh(
        self, actor_name, mesh=None, glyph_shape=None, glyph_kwargs=dict(),
        orient=None, scale=None, factor=None, **mesh_kwargs
    ):
        if mesh is None:
            mesh = self.fullmesh
        if glyph_shape is None:
            glyph_shape = self.geometric_objects["cylinder"](
                radius=0.5/self.settings["rod_aspect_ratio"],
                height=1
            )
        if orient is not None:
            glyph_kwargs["orient"] = orient
        if scale is not None:
            glyph_kwargs["scale"] = scale
        if factor is not None:
            glyph_kwargs["factor"] = factor
        glyph_kwargs["geom"] = glyph_shape
        glyph_kwargs["tolerance"] = None # forbid interpolation
        self.update_actor(
            actor_name,
            filter=mesh.glyph,
            filter_kwargs=glyph_kwargs,
            **mesh_kwargs
        )

    def relink_visibility_checkbox(self, actor_name):
        if actor_name in self.visibility_checkboxes.keys():
            checkbox = self.visibility_checkboxes[actor_name]
            checkbox.toggled.disconnect()
            checkbox.toggled.connect(
                lambda: self.toggle_visibility(actor_name)
            )
            checkbox.toggled.connect(self.set_checkbox_symbol(checkbox))

    def add_if_dict_lacks(self, to_dict, from_dict):
        for key in from_dict.keys():
            if not key in to_dict.keys():
                to_dict[key] = from_dict[key]

    def make_slice_widget_callback(self, actor_name=None, what_to_put_func=None, slice_name=None, mesh_to_slice=None):
        if mesh_to_slice is None:
            mesh_to_slice = self.coarsemesh
        if slice_name is None and actor_name is not None:
            slice_name = actor_name + "_plane"
        def slice_widget_callback(normal, origin):
            """make glyph plot and transparent plane for director field slice"""
            origin = tuple(
                self.director_resolution * np.asarray(
                    np.array(origin)/self.director_resolution, dtype=int
                )
            )
            slc = mesh_to_slice.slice(normal=normal, origin=origin)
            if actor_name is not None and what_to_put_func is not None:
                actor_dict = self.get_actor_dict(actor_name)
                self.add_if_dict_lacks(
                    actor_dict["mesh_kwargs"],
                    self.settings["default_mesh_kwargs"]
                )
                # for key in self.settings["default_mesh_kwargs"].keys():
                #     if not key in actor_dict["mesh_kwargs"].keys():
                #         actor_dict["mesh_kwargs"][key] = self.settings["default_mesh_kwargs"][key]
                try:
                    what_to_put_func(slc)
                except ValueError:
                    pass
            # self.add_glyphs_to_mesh(
            #     actor_name=actor_name,
            #     mesh=slc,
            #     glyph_shape=self.geometric_objects["cylinder"](
            #         radius=0.2, height=1,
            #         resolution=self.settings["cylinder_resolution"]
            #     ),
            #     orient="director", scale="nematic_sites",
            #     factor=self.director_resolution,
            #     **actor_dict["mesh_kwargs"]
            # )
            actor_dict = self.get_actor_dict(slice_name)
            self.add_if_dict_lacks(
                actor_dict["mesh_kwargs"],
                self.settings["default_slice_kwargs"]
            )
            # default_slice_kwargs = self.settings["default_slice_kwargs"]
            # for key in default_slice_kwargs:
            #     if not key in actor_dict["mesh_kwargs"].keys():
            #         actor_dict["mesh_kwargs"][key] = default_slice_kwargs[key]
            try:
                self.update_actor(
                    slice_name,
                    filter = lambda : slc,
                    **actor_dict["mesh_kwargs"]
                )
            except ValueError:
                pass
        return slice_widget_callback

    def make_director_slice_func(self):
        return self.make_slice_widget_callback(
            "director",
            lambda slc: self.add_glyphs_to_mesh(
                actor_name="director",
                mesh=slc,
                glyph_shape=self.geometric_objects["cylinder"](
                    radius=0.2, height=1,
                    resolution=self.settings["cylinder_resolution"]
                ),
                orient="director", scale="nematic_sites",
                factor=self.director_resolution,
                **self.actors_dict["director"]["mesh_kwargs"]
            ),
            slice_name="slice_plane"
        )
        # def director_slice_func(normal, origin):
        #     """make glyph plot and transparent plane for director field slice"""
        #     origin = tuple(
        #         self.director_resolution * np.asarray(
        #             np.array(origin)/self.director_resolution, dtype=int
        #         )
        #     )
        #     slc = self.coarsemesh.slice(normal=normal, origin=origin)
        #     actor_dict = self.get_actor_dict("director")
        #     for key in self.settings["default_mesh_kwargs"]:
        #         if not key in actor_dict["mesh_kwargs"].keys():
        #             actor_dict["mesh_kwargs"][key] = self.settings["default_mesh_kwargs"][key]
        #     self.add_glyphs_to_mesh(
        #         actor_name="director",
        #         mesh=slc,
        #         glyph_shape=self.geometric_objects["cylinder"](
        #             radius=0.2, height=1,
        #             resolution=self.settings["cylinder_resolution"]
        #         ),
        #         orient="director", scale="nematic_sites",
        #         factor=self.director_resolution,
        #         **actor_dict["mesh_kwargs"]
        #     )
        #     actor_dict = self.get_actor_dict("slice_plane")
        #     default_slice_kwargs = self.settings["default_slice_kwargs"]
        #     for key in default_slice_kwargs:
        #         if not key in actor_dict["mesh_kwargs"].keys():
        #             actor_dict["mesh_kwargs"][key] = default_slice_kwargs[key]
        #     self.update_actor(
        #         "slice_plane",
        #         filter = lambda : slc,
        #         **actor_dict["mesh_kwargs"]
        #     )
        # return director_slice_func

    def set_color(self, actor_name, color):
        if not actor_name in self.renderer.actors:
            actor_name += "_isosurface"
        self.colors[actor_name] = color
        try:
            self.update_actor(actor_name, color=color)
        except:
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

    def update_actor(self, actor_name, filter=None, filter_kwargs=None, dataset_name = None, **kwargs):

        actor_dict = self.get_actor_dict(actor_name)
        if filter is not None:
            actor_dict["filter"] = filter
        if type(filter_kwargs) is dict:
            for key in filter_kwargs.keys():
                actor_dict["filter_kwargs"][key] = filter_kwargs[key]
        if dataset_name is not None:
            actor_dict["dataset_name"] = dataset_name
        if actor_name in self.renderer.actors.keys():
            visibility = self.renderer.actors[actor_name].GetVisibility()
        else:
            visibility = 1

        if "color" in kwargs.keys(): # for newly passed color setting
            color = kwargs["color"]
        elif "color" in actor_dict["mesh_kwargs"].keys():
            # for previously set color
            color = actor_dict["mesh_kwargs"]["color"]
            if color is None and "scalars" in actor_dict["mesh_kwargs"].keys():
                color = actor_dict["mesh_kwargs"]["scalars"]
        else:
            color = None
        if color is None:
            if actor_name in self.colors.keys():
                color = self.colors[actor_name]
            else:
                color = self.settings["default_isosurface_color"]
        self.colors[actor_name] = color

        # sb_lut = sb_pos = sb_vis = sb_ht = sb_wt = None
        sb_vis = None
        if color in self.scalar_fields(): # "color" is really "scalars"
            kwargs["scalars"] = color
            # if not "show_scalar_bar" in contour_kwargs.keys():
            kwargs["show_scalar_bar"] = True
            if not "scalar_bar_args" in kwargs.keys():
                kwargs["scalar_bar_args"] = self.settings["scalar_bar_args"]

            new_scalar_bar_title = color
            # make sure we're not taking the name of another actor's scalar bar
            if "scalar_bar" in actor_dict.keys():
                old_scalar_bar_name = actor_dict["scalar_bar"].GetTitle()
                do_title_check = (
                    new_scalar_bar_title != old_scalar_bar_name
                )
                old_scalar_bar = self.scalar_bars[old_scalar_bar_name]
                # sb_lut = old_scalar_bar.GetLookupTable()
                # sb_pos = old_scalar_bar.GetPosition()
                sb_vis = old_scalar_bar.GetVisibility()
                # sb_ht = old_scalar_bar.GetHeight()
                # sb_wt = old_scalar_bar.GetWidth()
                scalar_bar_actor_name = self.get_scalar_bar_actor(
                    scalar_bar_name=old_scalar_bar_name
                )
                # if scalar_bar_actor_name in self.viscolor_toolbars.keys():
                #     self.viscolor_toolbars[scalar_bar_actor_name].destroy()
                self.remove_scalar_bar(old_scalar_bar_name)

            else:
                do_title_check = True
            if do_title_check:
                new_scalar_bar_title = self.name_without_overwriting(
                    new_scalar_bar_title,
                    self.scalar_bars
                )
                # i=0
                # while new_scalar_bar_title in self.scalar_bars.keys():
                #     i += 1
                #     new_scalar_bar_title = (
                #         try_new_scalar_bar_title + "_" + str(i)
                #     )


            kwargs["scalar_bar_args"]["title"] = new_scalar_bar_title
            kwargs["scalar_bar_args"]["title_font_size"] = self.settings["scalar_bar_args"]["title_font_size"]
            kwargs["color"] = None
        else: # "color" is actually a color
            kwargs["color"] = color
            kwargs["scalars"] = None
            kwargs["show_scalar_bar"] = False

            if "scalar_bar" in actor_dict.keys():
                old_scalar_bar_name = actor_dict["scalar_bar"].GetTitle()
                self.remove_scalar_bar(old_scalar_bar_name)
                scalar_bar_actor_name = self.get_scalar_bar_actor(
                    scalar_bar_name=old_scalar_bar_name
                )
                # if scalar_bar_actor_name in self.viscolor_toolbars.keys():
                #     self.viscolor_toolbars[scalar_bar_actor_name].destroy()
                if actor_name in self.colorbars.keys():
                    del self.colorbars[actor_name]
                del actor_dict["scalar_bar"]

        kwargs["name"] = actor_name
        for key in kwargs.keys():
            actor_dict["mesh_kwargs"][key] = kwargs[key]


        actor_dict["actor"] = self.add_mesh(
            actor_dict["filter"](**actor_dict["filter_kwargs"]),
            **actor_dict["mesh_kwargs"]
        )

        if actor_dict["mesh_kwargs"]["show_scalar_bar"]:
            scalar_bar = self.scalar_bars[
                actor_dict["mesh_kwargs"]["scalar_bar_args"]["title"]
            ]
            self.colorbars[actor_name] = scalar_bar.GetTitle()
            actor_dict["scalar_bar"] = scalar_bar
            # self.add_viscolor_toolbar_for_colorbar(actor_name)
            self.standardize_scalar_bar(scalar_bar)
            if sb_vis is not None:
                scalar_bar.SetVisibility(sb_vis)



        self.renderer.actors[actor_name].SetVisibility(visibility)
        self.relink_visibility_checkbox(actor_name)

    def get_actor_dict(self, actor_name):
        if not actor_name in self.actors_dict.keys():
            self.actors_dict[actor_name] = dict(
                mesh_kwargs = dict(
                    name=actor_name
                ),
                filter_kwargs = dict()
            )
        return self.actors_dict[actor_name]

    def update_isosurface(self, actor_name, dataset_name = None, contour_values=None, **contour_kwargs):

        actor_dict = self.get_actor_dict(actor_name)
        if dataset_name is None:
            dataset_name = actor_dict["dataset_name"]
        if contour_values is None:
            if not "contour_values" in actor_dict.keys():
                actor_dict["contour_values"] = [0.5]
            # else keep existing contour values
        else:
            if type(contour_values) is not list:
                contour_values = [contour_values]
            actor_dict["contour_values"] = contour_values
        self.update_actor(
            actor_name,
            self.fullmesh.contour,
            dict(
                isosurfaces=actor_dict["contour_values"],
                scalars=dataset_name
            ),
            **contour_kwargs
        )


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

        if mesh_color_scalars is not None:
            kwargs["color"] = mesh_color_scalars
        else:
            kwargs["color"] = color
        try:
            self.update_isosurface(actor_name, dataset_name=scalars, contour_values=[contour_value], **kwargs)

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
        self.toolbars[actor_name].setFixedWidth(self.settings["QSliders_toolbar_width"])
        self.app_window.addToolBar(
            Qt.Qt.LeftToolBarArea,
            self.toolbars[actor_name]
        )

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
            if actor_name in self.QSliders.keys():
                self.wiggle_slider_to_update(self.QSliders[actor_name])
            if actor_name in self.scalar_bars:
                self.scalar_bars[actor_name].SetVisibility(actor.GetVisibility())
        return return_function

    def set_lights_intensity(self, intensity):
        for light in self.renderer.lights:
            light.SetIntensity(intensity)

    def rescale_lights_intensity(self, factor):
        for light in self.renderer.lights:
            light.SetIntensity(factor*light.GetIntensity())

    def add_isosurface_slider_aux(self, scalars_name):
        def return_function():
            self.add_isosurface_slider(scalars_name)
        return return_function




    def name_without_overwriting(self, try_new_name, dict_of_old_names):
        new_name = try_new_name
        i=1
        while new_name in dict_of_old_names.keys():
            i += 1
            new_name = try_new_name + "_" + str(i)
        return new_name

    def add_slice_aux(self, scalars_name):
        def return_function():
            self.add_slice(scalars_name)
        return return_function

    def add_slice(self, scalars_name, slice_name=None, widget_name=None):
        if slice_name is None:
            slice_name = self.name_without_overwriting(
                scalars_name + "_slice",
                self.actors_dict
            )
            # try_slice_name = scalars_name+"_slice"
            # slice_name = try_slice_name
            # i=1
            # while slice_name in self.actors_dict.keys():
            #     i += 1
            #     slice_name = try_slice_name + "_" + str(i)
        if widget_name is None:
            widget_name = self.name_without_overwriting(
                slice_name + "_widget",
                self.widgets
            )
            # try_widget_name = slice_name + "_widget"
            # widget_name = try_widget_name
            # i=1
            # while widget_name in self.widgets.keys():
            #     i += 1
            #     widget_name = try_widget_name + "_" + str(i)

        self.setup_slice_widget(
            widget_name,
            self.make_slice_widget_callback(
                slice_name = slice_name,
                mesh_to_slice = self.fullmesh
            )
        )
        self.update_actor(
            slice_name,
            color = scalars_name,
            **self.settings["default_slice_kwargs"]
        )
        self.renderer.actors[slice_name].SetVisibility(True)

        if slice_name in self.colorbars.keys():
            scalar_bar = self.scalar_bars[self.colorbars[slice_name]]
            self.standardize_scalar_bar(scalar_bar)

        if not slice_name in self.toolbars.keys():
            self.add_viscolor_toolbar(
                slice_name,
                label=slice_name
            )
        #
        # def refresh_callback(normal=None, origin=None):
        #     slice_kwargs = dict(origin=origin)
        #     if normal is not None:
        #         slice_kwargs["normal"]=normal
        #     self.slices[slice_name] = self.fullmesh.slice(**slice_kwargs)
        #     self.update_actor(
        #         slice_name,
        #         lambda : self.slices[slice_name],
        #         **dict(
        #             color=scalars_name,
        #             ambient=1, specular=0, diffuse=0
        #         )
        #     )

        # self.widgets[widget_name] = self.add_plane_widget(
        #     refresh_callback,
        # )
        # self.refresh_functions[slice_name] = refresh_callback
        # self.refresh_functions[slice_name]()
        # self.standardize_scalar_bar(self.scalar_bars[scalars_name])
        # self.renderer.actors[slice_name].SetVisibility(True)
        # self.add_to_toggle_menu(slice_name)
        # self.add_plane_widget_to_widget_menu(widget_name)

    def standardize_scalar_bar(self, scalar_bar):
        scalar_bar.SetMaximumHeightInPixels(
            self.settings["scalar_bar_maxheight"]
        )
        scalar_bar.SetMaximumWidthInPixels(
            self.settings["scalar_bar_maxwidth"]
        )
        scalar_bar.SetTextPad(self.settings["scalar_bar_text_pad"])
        scalar_bar.UnconstrainedFontSizeOff()


    def get_scalar_bar_actor(self, scalar_bar_name=None, actor_name=None):
        if scalar_bar_name is None and actor_name is not None:
            scalar_bar_name = self.colorbars[actor_name]
        scalar_bar = self.scalar_bars[scalar_bar_name]
        for actor_name in self.renderer.actors.keys():
            if self.renderer.actors[actor_name] is scalar_bar:
                return actor_name

    def add_viscolor_toolbar_for_colorbar(self, actor_name):
        if actor_name in self.toolbars.keys() and actor_name in self.colorbars.keys():
            self.add_viscolor_toolbar(
                self.get_scalar_bar_actor(actor_name=actor_name),
                parent_toolbar = self.toolbars[actor_name],
                label='  colorbar',
                is_colorbar=True
            )

    def add_viscolor_toolbar(self, actor_name, label=None, parent_toolbar=None,
    is_colorbar=False):
        toolbar_row = qw.QToolBar()
        self.add_visibility_checkbox(actor_name, toolbar_row)
        # self.add_color_picker_button(actor_name, toolbar_row)
        if parent_toolbar is None:
            self.toolbars[actor_name] = qw.QToolBar(actor_name)
            parent_toolbar = self.toolbars[actor_name]
            self.app_window.addToolBar(
                Qt.Qt.LeftToolBarArea,
                parent_toolbar
            )
        if label is not None:
            parent_toolbar.addWidget(qw.QLabel(label))
        parent_toolbar.addWidget(toolbar_row)
        self.viscolor_toolbars[actor_name] = toolbar_row
        self.add_color_options_menu(actor_name)


    def color_picker(self, actor_name):
        self.set_color(actor_name, qw.QColorDialog.getColor().name())


    def add_color_picker_button(self, actor_name, toolbar):
        color_button = qw.QPushButton('🎨')
        color_button.setToolTip('choose solid color for '+actor_name)
        color_button.setMaximumWidth(45)
        color_button.clicked.connect(lambda: self.color_picker(actor_name))
        toolbar.addWidget(color_button)

    def add_visibility_checkbox(self, actor_name, toolbar):
        checkbox = qw.QCheckBox(self.settings["visible_symbol"])
        checkbox.setChecked(self.renderer.actors[actor_name].GetVisibility())
        checkbox.toggled.connect(self.set_checkbox_symbol(checkbox))
        checkbox.toggled.connect(lambda : self.toggle_visibility(actor_name))
        checkbox.setToolTip('toggle visibility of \"'+actor_name + '\"')
        toolbar.addWidget(checkbox)
        self.visibility_checkboxes[actor_name] = checkbox

    def add_color_options_menu(self, actor_name):
        minwidth=300
        actor_dict = self.actors_dict[actor_name]
        if "scalars" in actor_dict["mesh_kwargs"].keys():
            current_color_array_name = actor_dict["mesh_kwargs"]["scalars"]
        else:
            current_color_array_name = None
        if "cmap" in actor_dict["mesh_kwargs"].keys():
            current_cmap = actor_dict["mesh_kwargs"]["cmap"]
        else:
            current_cmap = None
        color_picker_choice = lambda: self.color_picker(actor_name)

        cb_colorby = qw.QComboBox()
        cb_colorby.addItems(self.scalar_fields())
        if current_color_array_name in self.scalar_fields():
            cb_colorby.setCurrentIndex(self.scalar_fields().index(current_color_array_name))
        else:
            cb_colorby.setCurrentIndex(-1)

        def update_clim_maxmin_display():
            if self.colors[actor_name] in self.scalar_fields():
                if "clim" in self.actors_dict[actor_name]["mesh_kwargs"]:
                    vmin, vmax = self.actors_dict[actor_name]["mesh_kwargs"]["clim"]
                else:
                    scalar_field = self.colors[actor_name]
                    vmin = np.min(self.fullmesh[scalar_field])
                    vmax = np.max(self.fullmesh[scalar_field])
                    self.update_actor(actor_name, clim=[vmin,vmax])

                sb_clim_min.setValue(vmin)
                sb_clim_max.setValue(vmax)

        def colorby_callback(choice_num):
            color_array_name = self.scalar_fields()[choice_num]
            vmin = np.min(self.fullmesh[color_array_name])
            vmax = np.max(self.fullmesh[color_array_name])
            self.update_actor(
                actor_name,
                color=color_array_name,
                clim=[vmin,vmax]
            )
            update_clim_maxmin_display()

        cb_colorby.currentIndexChanged.connect(colorby_callback)
        cb_cmap = qw.QComboBox()
        colormaps = plt.colormaps()
        cb_cmap.addItems(colormaps)
        if current_cmap in colormaps:
            cb_cmap.setCurrentIndex(colormaps.index(current_cmap))
        else:
            cb_cmap.setCurrentIndex(-1)
        cb_cmap.currentIndexChanged.connect(
            lambda choice_num: self.update_actor(
                actor_name,
                cmap=colormaps[choice_num]
            )
        )
        sb_clim_min = qw.QDoubleSpinBox()
        sb_clim_min.valueChanged.connect(
            lambda value: self.update_actor(
                actor_name,
                clim=[
                    min(value, sb_clim_max.value()),
                    sb_clim_max.value()
                ]
            )
        )
        sb_clim_min.setSingleStep(0.1)
        sb_clim_max = qw.QDoubleSpinBox()
        sb_clim_max.valueChanged.connect(
            lambda value: self.update_actor(
                actor_name,
                clim=[
                    sb_clim_min.value(),
                    max(sb_clim_min.value(), value)
                ]
            )
        )
        sb_clim_max.setSingleStep(0.1)

        color_array_widget = qw.QWidget()
        color_array_widget.setWindowTitle(f'Color for \"{actor_name}\"')
        layout = qw.QVBoxLayout(color_array_widget)
        formlayout = qw.QFormLayout()
        formlayout.addRow(qw.QLabel('color array'), cb_colorby)
        formlayout.addRow(qw.QLabel('colormap'), cb_cmap)
        formlayout.addRow(qw.QLabel('min value'), sb_clim_min)
        formlayout.addRow(qw.QLabel('max value'), sb_clim_max)
        layout.addLayout(formlayout)

        def color_array_choice():
            update_clim_maxmin_display()
            color_array_widget.show()

        set_opacity_widget = qw.QWidget()
        set_opacity_widget.setWindowTitle(f'Set opacity for \"{actor_name}\"')
        sb_set_opacity = qw.QDoubleSpinBox(minimum=0, maximum=1)
        sb_set_opacity.setSingleStep(0.1)
        sb_set_opacity.valueChanged.connect(
            lambda value: self.update_actor(actor_name, opacity=value)
        )
        opacity_widget_layout = qw.QVBoxLayout(set_opacity_widget)
        opacity_form_layout = qw.QFormLayout()
        opacity_form_layout.addRow(qw.QLabel('opacity: '), sb_set_opacity)
        opacity_widget_layout.addLayout(opacity_form_layout)
        set_opacity_widget.setMinimumWidth(minwidth)

        def set_opacity_choice():
            if "opacity" in self.actors_dict[actor_name]['mesh_kwargs'].keys():
                opacity = self.actors_dict[actor_name]['mesh_kwargs']['opacity']
            else:
                opacity = 1
            sb_set_opacity.setValue(opacity)
            set_opacity_widget.show()


        def toggle_colorbar_choice():
            if actor_name in self.colorbars.keys():
                colorbar = self.scalar_bars[self.colorbars[actor_name]]
                colorbar.SetVisibility(1-colorbar.GetVisibility())

        cb_children = [
            lambda: None, color_picker_choice, color_array_choice, toggle_colorbar_choice, set_opacity_choice
        ]
        cb_parent = qw.QComboBox()
        cb_parent.setMaximumWidth(60)
        cb_parent.addItems(["🎨", "solid color", "color array", "show/hide colorbar", "set opacity"])
        cb_parent.setCurrentIndex(0)
        cb_parent.setToolTip('Color options for \"' + actor_name + "\"")
        def index_change_callback(choice_num):
            cb_children[choice_num]()
            cb_parent.setCurrentIndex(0)

        cb_parent.currentIndexChanged.connect(index_change_callback)

        self.viscolor_toolbars[actor_name].addWidget(cb_parent)





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
