"""
Provides the :class:`BackendBaseMeta` class.
"""
# pylint: disable=bad-mcs-classmethod-argument


class BackendBaseMeta(type):
    """Metaclass for :class:`BackendBase` to automatically detected available
    and implemented shape methods.
    """
    def __new__(mcs, name, bases, attrs):  # **kwargs
        def get_abstract_shape_methods(method_names, attrs_):
            abstract_shape_methods_ = {}

            for method_name in method_names:
                if not method_name.startswith('__'):
                    is_abstract_shape_method = getattr(
                        attrs[method_name], '_abstract_shape_method', False
                    )
                    if is_abstract_shape_method:
                        shape_type = getattr(
                            attrs[method_name], '_shape_type', False
                        )

                        # TODO: do not use __name__ here
                        abstract_shape_methods_[shape_type] = attrs_[method_name].__name__

            return abstract_shape_methods_

        def get_implemented_shape_methods(method_names, attrs_, abstract_shape_methods_):
            # would be nicer: https://pypi.org/project/bidict/
            abstract_shape_methods_rev = dict(map(reversed, abstract_shape_methods_.items()))
            implemented_shapes = {}
            for method_name in method_names:
                if method_name in abstract_shape_methods_.values():
                    is_abstract_shape_method = getattr(
                        attrs_[method_name], '_abstract_shape_method', False
                    )

                    if not is_abstract_shape_method:
                        implemented_shapes[
                            abstract_shape_methods_rev[method_name]
                        ] = attrs_[method_name]

            return implemented_shapes

        # available shapes
        abstract_shape_methods = {}

        for base in bases:
            abstract_shape_methods.update(
                base._abstract_shape_methods
                # get_abstract_shape_methods(
                #     [attr for attr, _ in inspect.getmembers(base)],
                #     base.__dict__
                # )
            )

        abstract_shape_methods.update(
            get_abstract_shape_methods(attrs.keys(), attrs)
        )

        attrs['_abstract_shape_methods'] = abstract_shape_methods

        # implemented shapes
        implemented_shape_methods = {}

        for base in bases:
            implemented_shape_methods.update(
                base._implemented_shape_methods
                # get_implemented_shape_methods(
                #     [attr for attr, _ in inspect.getmembers(base)],
                #     base.__dict__,
                #     abstract_shape_methods
                # )
            )

        implemented_shape_methods.update(
            get_implemented_shape_methods(attrs.keys(), attrs, abstract_shape_methods)
        )

        attrs['_implemented_shape_methods'] = implemented_shape_methods

        return super().__new__(mcs, name, bases, attrs)
