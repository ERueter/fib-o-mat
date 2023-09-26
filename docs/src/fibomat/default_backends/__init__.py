"""
This submodule implements two exemplary backends.
The first one, the bokeh backend, can be used to visualize a project. It uses the bokeh library to create a plot which
can be viewed with a web browser.
The second one, the spotlist backend, rasterizes all shapes and creates a list of dwell points and times.
"""


from fibomat.backend import registry

from fibomat.default_backends.bokeh_backend import BokehBackend, StubRasterStyle
# from fibomat.default_backends.donothing_backend import DoNothingBackend
from fibomat.default_backends.spotlist_backend import SpotListBackend
from fibomat.default_backends.svg_backend import SVGBackend
from fibomat.default_backends.bitmap_backend import BitmapBackend


registry.register(BokehBackend, BokehBackend.name)
registry.register(SpotListBackend, SpotListBackend.name)
# registry.register(DoNothingBackend, DoNothingBackend.name)

__all__ = ['BokehBackend', 'StubRasterStyle', 'SpotListBackend']
