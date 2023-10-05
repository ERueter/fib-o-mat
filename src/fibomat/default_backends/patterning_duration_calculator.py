import numpy as np
from prettytable import PrettyTable


from fibomat.pattern import Pattern
from fibomat.units import Q_
from fibomat.site import Site

from fibomat.backend.backendbase import BackendBase

from fibomat.shapes import (
    Ellipse,
    Ring,
)
from fibomat.raster_styles import two_d, one_d


class PatterningDurationCalculator(BackendBase):
    name = "PatterningDurationCalculator"

    def __init__(self, current: Q_, *args, **kwargs):
        self._current = current
        super().__init__(*args, **kwargs)

        self.durations = []
        self._i_site = 0
        self._i_pattern = 0

    def _add_pattern(self, ptn, duration):
        self.durations[-1]["patterns"].append(
            {
                "name": f"{self._i_pattern} {ptn.dim_shape.shape.__class__.__name__}",
                "duration": duration,
            }
        )

    def print(self):
        def _format_time(t):
            if t < Q_("60 s"):
                t.ito("s")
            elif t < Q_("60 min"):
                t.ito("min")
            else:
                t.ito("hours")
            return f"{t:~P.2f}"

        def _percent_to_star(p):
            n = 20
            n_stars = int(p * n)
            return "*" * n_stars + " " * (n - n_stars)

        table = PrettyTable()
        table.field_names = [
            "Site",
            "#Patterns",
            "Duration",
            "Cum. duration",
            "Rel. duration",
        ]
        table.align["Site"] = "l"
        table.align["#Patterns"] = "r"
        table.align["Duration"] = "r"
        table.align["Cum. duration"] = "r"
        table.align["Rel. duration"] = "r"

        total_duration = 0

        durations_per_site = []

        for site in self.durations:
            duration = sum([pattern["duration"] for pattern in site["patterns"]])
            durations_per_site.append(duration)
            total_duration += duration

        cum_duration = 0
        for site, duration in zip(self.durations, durations_per_site):
            cum_duration += duration
            rel_duration = float(duration / total_duration)
            table.add_row(
                [
                    site["name"],
                    len(site["patterns"]),
                    _format_time(duration),
                    _format_time(cum_duration),
                    f">|{_percent_to_star(rel_duration)}|< ({rel_duration:.3f})",
                ]
            )

        print(table)
        print()
        print(
            f'Total duration: {_format_time(total_duration)} @ {self._current.to("pA"):~P.2f}'
        )

    def process_site(self, new_site: Site) -> None:
        # self.dwell_times_per_site.append(0.)

        name = f"Site {self._i_site}"

        if descr := new_site.description:
            name += f" {descr}"

        self.durations.append({"name": name, "patterns": []})

        self._i_site += 1
        self._i_pattern = 0

        return super().process_site(new_site)

    def polygon(self, ptn) -> None:
        # print('warning: polygon not counted')
        # https://stackoverflow.com/a/49129646
        def polygon_area(x, y):
            correction = x[-1] * y[0] - y[-1] * x[0]
            main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
            return 0.5 * np.abs(main_area + correction)

        if isinstance(ptn.raster_style, two_d.LineByLine) and isinstance(
            ptn.raster_style.line_style, one_d.Curve
        ):
            mill = ptn.mill
            pitch = ptn.raster_style.line_pitch * ptn.raster_style.line_style.pitch

            dwell_time = mill["dwell_time"]
            try:
                repeats = mill["repeats"]
            except KeyError:
                # print('dose', mill['dose'], mill._kwargs)
                repeats = mill["dose"] * pitch / (dwell_time * self._current)
            area = polygon_area(
                ptn.dim_shape.shape.points[:, 0], ptn.dim_shape.shape.points[:, 1]
            )
            time = area * ptn.dim_shape.unit**2 / pitch * repeats * dwell_time

            self._add_pattern(ptn, time)
        else:
            raise NotImplementedError

    def arc_spline(self, ptn) -> None:
        """
        Adds pattern with `Line` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Line` as shape

        Returns:
            None
        """
        if isinstance(ptn.raster_style, one_d.Curve):
            mill = ptn.mill
            pitch = ptn.raster_style.pitch
            dwell_time = mill["dwell_time"]
            try:
                repeats = mill["repeats"]
            except KeyError:
                # print('dose', mill['dose'], mill._kwargs)
                repeats = mill["dose"] * pitch / (dwell_time * self._current)

            time = (
                ptn.dim_shape.shape.length
                * ptn.dim_shape.unit
                / ptn.raster_style.pitch
                * repeats
                * dwell_time
            )
            self._add_pattern(ptn, time)
        else:
            print(ptn.raster_style)
            raise NotImplementedError

    def _process_as_arc_spline_pattern(self, ptn):
        arc_spline = ptn.dim_shape.shape.to_arc_spline()

        return self.arc_spline(
            Pattern(
                dim_shape=arc_spline * ptn.dim_shape.unit,
                raster_style=ptn.raster_style,
                mill=ptn.mill,
            )
        )

    def line(self, ptn: Pattern) -> None:
        """
        Adds pattern with `Line` as shape to the backend.

        Args:
            ptn (Pattern): pattern with `Line` as shape

        Returns:
            None
        """
        return self._process_as_arc_spline_pattern(ptn)

    def rect(self, ptn: Pattern) -> None:
        if isinstance(ptn.raster_style, two_d.LineByLine) and isinstance(
            ptn.raster_style.line_style, one_d.Curve
        ):
            mill = ptn.mill
            pitch = ptn.raster_style.line_pitch * ptn.raster_style.line_style.pitch

            dwell_time = mill["dwell_time"]
            try:
                repeats = mill["repeats"]
            except KeyError:
                # print('dose', mill['dose'], mill._kwargs)
                repeats = mill["dose"] * pitch / (dwell_time * self._current)

            time = (
                ptn.dim_shape.shape.width
                * ptn.dim_shape.shape.height
                * ptn.dim_shape.unit**2
                / pitch
                * repeats
                * dwell_time
            )

            self._add_pattern(ptn, time)
        else:
            raise NotImplementedError

    def polyline(self, ptn: Pattern) -> None:
        return self._process_as_arc_spline_pattern(ptn)

    def circle(self, ptn: Pattern) -> None:
        ellipse = Ellipse(
            ptn.dim_shape.shape.r,
            ptn.dim_shape.shape.r,
            center=ptn.dim_shape.shape.center,
        )

        return self.ellipse(
            Pattern(
                dim_shape=ellipse * ptn.dim_shape.unit,
                raster_style=ptn.raster_style,
                mill=ptn.mill,
            )
        )

    def ellipse(self, ptn: Pattern) -> None:
        if isinstance(ptn.raster_style, two_d.LineByLine) and isinstance(
            ptn.raster_style.line_style, one_d.Curve
        ):
            mill = ptn.mill
            pitch = ptn.raster_style.line_pitch * ptn.raster_style.line_style.pitch

            dwell_time = mill["dwell_time"]
            try:
                repeats = mill["repeats"]
            except KeyError:
                # print('dose', mill['dose'], mill._kwargs)
                repeats = mill["dose"] * pitch / (dwell_time * self._current)

            area = np.pi * ptn.dim_shape.shape.a * ptn.dim_shape.shape.b

            time = area * ptn.dim_shape.unit**2 / pitch * repeats * dwell_time

            self._add_pattern(ptn, time)
        else:
            raise NotImplementedError

    def ring(self, ptn: Pattern[Ring]):
        if isinstance(ptn.raster_style, two_d.LineByLine) and isinstance(
            ptn.raster_style.line_style, one_d.Curve
        ):
            mill = ptn.mill
            pitch = ptn.raster_style.line_pitch * ptn.raster_style.line_style.pitch

            dwell_time = mill["dwell_time"]
            try:
                repeats = mill["repeats"]
            except KeyError:
                # print('dose', mill['dose'], mill._kwargs)
                repeats = mill["dose"] * pitch / (dwell_time * self._current)

            area = np.pi * (
                ptn.dim_shape.shape._r_outer**2
                - (ptn.dim_shape.shape._r_outer - ptn.dim_shape.shape._thickness) ** 2
            )

            time = area * ptn.dim_shape.unit**2 / pitch * repeats * dwell_time

            self._add_pattern(ptn, time)
        else:
            raise NotImplementedError

    def spot(self, ptn: Pattern) -> None:
        mill = ptn.mill

        try:
            dose = mill["dose"]
        except KeyError:
            dose = mill["dwell_time"] * mill["repeats"] * self._current
            # print('dose', mill['dose'], mill._kwargs)
        time = dose / self._current
        self._add_pattern(ptn, time)
