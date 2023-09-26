from fibomat.backend.backendbase import BackendBase


class DoNothingBackend(BackendBase):
    """This backend is only for testing purpose."""

    name = 'donothing'

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super().__init__()

    def null_shape(self, shape, mill, shape_units, **kwargs):
        self.null_shape = (shape, mill, shape_units, kwargs)  # noqa

