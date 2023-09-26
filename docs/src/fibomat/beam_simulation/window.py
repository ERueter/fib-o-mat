import enum

import numpy as np

from PyQt5 import QtWidgets, QtCore


from fibomat.beam_simulation.fibcanvas import FIBCanvas
from fibomat.beam_simulation.slider import Slider
from fibomat.beam_simulation.dwell_time_widget import DwellTimeWidget


@enum.unique
class AnimationState(enum.Enum):
    STARTED = enum.auto()
    STOPPED = enum.auto()
    SLIDER_STOPPED = enum.auto()


class AnimationWindow(QtWidgets.QMainWindow):
    def __init__(self, dwell_points: np.ndarray, length_unit: str, *args, **kwargs):
        super(AnimationWindow, self).__init__(*args, **kwargs)

        self._dwell_points = dwell_points

        self._update_slider_pos = True
        self._anim_state = AnimationState.STOPPED

        self._fib_canvas = FIBCanvas(dwell_points[:, :2], length_unit, self._animation_step)

        # self._dwell_time_widget = DwellTimeWidget(self._dwell_points[:, 2], self)

        self._init_gui()
        self._connect_signals()

    def _init_gui(self):
        self.setWindowTitle('Ion beam simulation')
        self.resize(800, 800)

        layout = QtWidgets.QVBoxLayout()

        self._fib_canvas.create_native()
        self._fib_canvas.native.setParent(self)
        # self._fib_canvas.native.resize(800, 600)
        self._fib_canvas.native.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        layout.addWidget(self._fib_canvas.native)

        # layout.addWidget(self._dwell_time_widget)

        layout.addWidget(QtWidgets.QLabel('Progress'))
        slider = Slider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(len(self._dwell_points))
        slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        slider.setTickInterval(int(len(self._dwell_points) / 10))
        layout.addWidget(slider)
        self._slider = slider

        self._start_stop_button = QtWidgets.QPushButton('Start')
        self._reset_button = QtWidgets.QPushButton('Reset')

        control_widget_layout = QtWidgets.QHBoxLayout()
        control_widget_layout.addWidget(self._start_stop_button)
        control_widget_layout.addWidget(self._reset_button)

        control_widget = QtWidgets.QWidget()
        control_widget.setLayout(control_widget_layout)
        layout.addWidget(control_widget)

        speed_label = QtWidgets.QLabel('Speed (points / s)')
        speed_label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignRight)
        self._speed_input = QtWidgets.QDoubleSpinBox()
        self._speed_input.setMinimum(.1)
        self._speed_input.setMaximum(len(self._dwell_points))
        self._speed_input.setValue(50)
        self._speed_input.setSingleStep(50)

        tail_label = QtWidgets.QLabel('Tail length')
        tail_label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignRight)
        self._tail_input = QtWidgets.QSpinBox()
        self._tail_input.setMinimum(1)
        self._tail_input.setMaximum(len(self._dwell_points) - 1)
        self._tail_input.setValue(100 if 100 < len(self._dwell_points) else len(self._dwell_points))
        self._tail_input.setSingleStep(50)

        alpha_label = QtWidgets.QLabel('Alpha')
        alpha_label.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignRight)
        self._alpha_input = QtWidgets.QDoubleSpinBox()
        self._alpha_input.setMinimum(.01)
        self._alpha_input.setMaximum(1)
        self._alpha_input.setValue(.2)
        self._alpha_input.setSingleStep(0.01)

        settings_widget_layout = QtWidgets.QHBoxLayout()
        settings_widget_layout.addWidget(speed_label)
        settings_widget_layout.addWidget(self._speed_input)
        settings_widget_layout.addWidget(tail_label)
        settings_widget_layout.addWidget(self._tail_input)
        settings_widget_layout.addWidget(alpha_label)
        settings_widget_layout.addWidget(self._alpha_input)

        settings_widget = QtWidgets.QWidget()
        settings_widget.setLayout(settings_widget_layout)
        layout.addWidget(settings_widget)

        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def _connect_signals(self):
        self._slider.sliderPressed.connect(self._slider_pressed)
        self._slider.sliderReleased.connect(self._slider_released)
        self._slider.valueChanged.connect(self._slider_value_changed)

        self._start_stop_button.clicked.connect(self._start_stop_clicked)
        self._reset_button.clicked.connect(self._reset_clicked)

        self._speed_input.valueChanged.connect(self._speed_value_changed)
        self._tail_input.valueChanged.connect(self._tail_value_changed)
        self._alpha_input.valueChanged.connect(self._alpha_value_changed)

        # set the default values
        self._speed_input.valueChanged.emit(self._speed_input.value())
        self._tail_input.valueChanged.emit(self._tail_input.value())
        self._alpha_input.valueChanged.emit(self._alpha_input.value())

    def _alpha_value_changed(self, alpha):
        self._fib_canvas.set_alpha(alpha)
        if self._anim_state.STOPPED:
            self._fib_canvas.update()

    def _tail_value_changed(self, length):
        self._fib_canvas.set_tail_length(length + 1)
        if self._anim_state.STOPPED:
            self._fib_canvas.update()

    def _speed_value_changed(self, speed):
        self._fib_canvas.set_speed(speed)

    def _slider_pressed(self):
        self._stop_animation()
        if self._anim_state == AnimationState.STARTED:
            self._anim_state = AnimationState.SLIDER_STOPPED

    def _slider_released(self):
        self._update_slider_pos = True

    def _slider_value_changed(self, value):
        self._fib_canvas.current_point_index = value
        if self._anim_state == AnimationState.SLIDER_STOPPED:
            self._fib_canvas.start_animation()
        else:
            self._fib_canvas.update()

    def _animation_step(self, i_step: int) -> bool:
        if i_step > self._fib_canvas.number_of_points:
            self._fib_canvas.current_point_index = self._fib_canvas.number_of_points - 1
            self._stop_animation()
            self._slider.setValue(self._fib_canvas.number_of_points - 1)

            return False
        else:
            if self._update_slider_pos:
                # self._dwell_time_widget.draw(i_step)
                self._slider.setValue(i_step)

            return True

    def _start_animation(self):
        self._anim_state = AnimationState.STARTED
        self._start_stop_button.setText('Stop')
        self._fib_canvas.start_animation()

    def _stop_animation(self):
        self._anim_state = AnimationState.STOPPED
        self._start_stop_button.setText('Start')
        self._fib_canvas.stop_animation()

    def _start_stop_clicked(self):
        if self._anim_state == AnimationState.STOPPED:
            self._start_animation()
        else:
            self._stop_animation()

    def _reset_clicked(self):
        if self._anim_state == AnimationState.STARTED:
            self._stop_animation()

        self._fib_canvas.reset_animation()
        self._slider.setValue(0)
