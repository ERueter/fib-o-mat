from typing import Callable

import numpy as np

from vispy import app
from vispy.visuals.transforms import STTransform

from fibomat.beam_simulation.markers import Tail, DwellPoints
from fibomat.beam_simulation.scalebar import ScaleBar


class FIBCanvas(app.Canvas):

    def __init__(self, dwell_points: np.ndarray, length_unit: str, on_timer_callback: Callable):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))

        dwell_points = np.asarray(dwell_points, dtype=np.float32)

        min_x, min_y = np.min(dwell_points, axis=0)
        max_x, max_y = np.max(dwell_points, axis=0)

        if self.physical_size[0] / (max_x - min_x) < self.physical_size[1] / (max_y - min_y):
            self._pre_scale = self.physical_size[0] / (max_x - min_x)
        else:
            self._pre_scale = self.physical_size[1] / (max_y - min_y)

        dwell_points *= self._pre_scale

        self._min_x, self._min_y = np.min(dwell_points, axis=0)
        self._max_x, self._max_y = np.max(dwell_points, axis=0)

        self._n_points = len(dwell_points)

        self._tail = Tail(dwell_points, 100, 10)
        self._dwell_points = DwellPoints(dwell_points)
        self._scale_bar = ScaleBar(length_unit, self)

        w, h = self.size
        self._dwell_points.transform = STTransform(translate=(w / 2., h / 2.))
        self._tail.transform = STTransform(translate=(w / 2., h / 2.))

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self._current_point = 0
        self._last_update = 0
        self._speed = 10
        self._run_animation = False

        self._on_timer_callback = on_timer_callback

    def on_resize(self, event):
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)

        self._dwell_points.transforms.configure(canvas=self, viewport=vp)
        self._dwell_points.transform.translate = self.physical_size[0] / 2, self.physical_size[1] / 2

        self._tail.transforms.configure(canvas=self, viewport=vp)
        self._tail.transform.translate = self.physical_size[0] / 2, self.physical_size[1] / 2

        width = self._max_x - self._min_x
        height = self._max_y - self._min_y

        margin = 0  # 10

        if self.physical_size[0] / (self._max_x - self._min_x) < self.physical_size[1] / (self._max_y - self._min_y):
            s = self.physical_size[0] / (width+margin)
            pixel_per_length = (self._max_x - self._min_x) / (self.physical_size[0] - 2*margin)
        else:
            s = self.physical_size[1] / (height+margin)
            pixel_per_length = (self._max_y - self._min_y) / (self.physical_size[1] - 2*margin)

        self._dwell_points.transform.scale = s, s
        self._tail.transform.scale = s, s
        self._scale_bar.transform(pixel_per_length / self._pre_scale, s)

    def on_draw(self, event):
        self.context.clear('white')
        self._dwell_points.draw()
        self._tail.draw(self._current_point)
        self._scale_bar.draw()

    def on_timer(self, event):
        if self._run_animation:
            freq = 1 / self._speed

            elapsed = self._timer.elapsed
            delta = elapsed - self._last_update

            if delta > freq:
                n_new_points = int(delta / freq)
                self._current_point += n_new_points
                self._last_update = elapsed
                self._on_timer_callback(self._current_point)
        self.update()

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

    def set_speed(self, speed):
        self._speed = speed

    def set_tail_length(self, length):
        self._tail._tail_length = length

    def set_alpha(self, alpha):
        self._tail._alpha = alpha

    @property
    def number_of_points(self) -> int:
        return self._n_points

    @property
    def current_point_index(self) -> int:
        return self._current_point

    @current_point_index.setter
    def current_point_index(self, value):
        self._current_point = value
