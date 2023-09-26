
import numpy as np

from PyQt5 import QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


# https://github.com/vispy/vispy/blob/af676d184be4aa778676963f054946d12e1b3ef9/examples/basics/scene/isocurve_for_trisurface_qt.py
class DwellTimeWidget(QtWidgets.QWidget):
    def __init__(self, dwell_times, parent):
        super().__init__(parent)

        self._dwell_times = dwell_times

        self.figure = plt.figure()
        self.figure.clear()

        ax = self.figure.add_subplot(111)

        ax.set_xlim(0, len(dwell_times)-1)
        ax.set_ylim(min(dwell_times), max(dwell_times))
        ax.set_xticks([])
        ax.tick_params(axis="y", direction="in", pad=-40)

        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_color('lightgrey')

        ax.yaxis.grid(color='lightgrey')

        plt.subplots_adjust(left=0, right=1, wspace=0, hspace=0) # bottom=0.0, right=1, top=1,

        self.dwell_time_canvas = FigureCanvas(self.figure)
        self.dwell_time_canvas.setGeometry(self.rect())

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.dwell_time_canvas)
        self.setLayout(layout)
        self.setFixedHeight(200)

        # ax.plot(dwell_times, '-', color='black')
        self.current_point, = ax.plot([], [], 'ro')


        self.dwell_time_canvas.draw()

        self.axbackground = self.figure.canvas.copy_from_bbox(ax.bbox)

        self.ax = ax

        # plt.show(block=False)

        # refresh canvas

    def draw(self, index: int):
        self.current_point.set_xdata([index])
        self.current_point.set_ydata([self._dwell_times])

        self.figure.canvas.restore_region(self.axbackground)
        self.ax.draw_artist(self.current_point)
        self.figure.canvas.blit(self.ax.bbox)

        self.figure.canvas.flush_events()

        #self.dwell_time_canvas.draw()

    def start_animation(self):
        # self.timer.start()
        self._run_animation = True
        self._last_update = self._timer.elapsed

    def stop_animation(self):
        # self.timer.stop()
        self._run_animation = False

    def reset_animation(self):
        self._last_update = 0
        self._current_point = 0
        self._run_animation = False
