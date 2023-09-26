from bokeh.models import Tool
from bokeh.core.properties import String, DashPattern, Color, Float


class MeasureTool(Tool):
    """
    A measure tool for bokeh plots!
    """

    # https://github.com/bokeh/bokeh/issues/9412
    __view_module__ = "bokeh-measuretool"

    # __javascript__ = "bokeh-measuretool.js"

    measure_unit = String(default="", help="")
    line_dash = DashPattern(default="solid", help="")
    line_color = Color(default="black", help="")
    line_width = Float(default=1, help="")
    line_alpha = Float(default=1.0, help="")
