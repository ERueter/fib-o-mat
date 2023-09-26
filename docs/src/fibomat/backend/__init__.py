"""
Backend related classes and functions.
Backends are responsible for exporting projects to a appropriate file a microscope (software) can understand.

See default_backend module for example backends.

if you Implement your own backend, make sure to register it with ::

    from fibomat.backend import registry, BackendBase

    class MyNewBackend(BackendBase):
        name = 'MyFancyName'
        ...

    registry.register(MyNewBackend, MyNewBackend.name)

The new backend can now be used within the `Project.export` method.

"""

from fibomat.backend.registry import registry
from fibomat.backend.backendbase import BackendBase

__all__ = ['registry', 'BackendBase']
