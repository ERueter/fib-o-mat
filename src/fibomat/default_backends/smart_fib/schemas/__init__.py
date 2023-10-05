import datetime as dt

from marshmallow import fields, pre_dump

from custom_backends.utils import OrderedSchema, LCBool, EnvelopeBaseSchema

class ElyVersionSchema(OrderedSchema):
    def _make_default_datetime(self, obj):
        return dt.datetime.now().strftime('%d.%m.%Y %H:%M:%S')

    created = fields.Method('_make_default_datetime', data_key='@created')
    modified = fields.Method('_make_default_datetime', data_key='@modified')
    number = fields.Str(data_key='@number', default="1.0")


class ElyAxesSchema(OrderedSchema):
    show = LCBool(data_key='@show', default=True)


class ElyGridSchema(OrderedSchema):
    horizontal = fields.Str(data_key='@horizontal', default='1.0')
    show = LCBool(data_key='@show', default=True)
    snap_to = LCBool(data_key='@snap_to', default=False)
    vertical = fields.Str(data_key='@vertical', default='1.0')


class ElyLayerSchema(OrderedSchema):
    fill_color = fields.Str(data_key='@fill_color', default='#00FF00')
    fill_opacity = fields.Str(data_key='@fill_opacity', default='0.5')
    hidden = LCBool(data_key='@hidden', default=False)
    locked = LCBool(data_key='@locked', default=False)
    name = fields.Str(data_key='@name')


class ElyLayerListSchema(OrderedSchema):
    layers = fields.List(fields.Nested(ElyLayerSchema), data_key='LAYER')


class ElyInstanceList(OrderedSchema):
    pass


class ElyGISSchema(OrderedSchema):
    name = fields.Str(data_key='@name', default='unchanged')
    channel = fields.Int(data_key='@channel', default=0)
    category = fields.Int(data_key='@category', default=0)
    type_ = fields.Str(data_key='@type', default='unchanged')
    ack = LCBool(data_key='@ack', default=False)
    autopark = LCBool(data_key='@autopark', default=False)
    offset = LCBool(data_key='@offset', default=False)
    usegas = LCBool(data_key='@usegas', default=False)


class ElyProbeSchema(OrderedSchema):
    name = fields.Str(data_key='@name')
    type_ = fields.Str(data_key='@type', default='specific')
    current = fields.Str(data_key='@current')
    diameter = fields.Str(data_key='@diameter')


# only working for spots!
class ElyExposureSchema(OrderedSchema):
    version = fields.Str(data_key='@version', default='2.1')
    column_type = fields.Str(data_key='@column_type', default='FIB')
    computed_parameter = fields.Str(data_key='@computed_parameter', default='by purpose')

    dwell_times_point = fields.Str(data_key='@dwell_times_point', default='1e-006 s')
    dwell_times_line = fields.Str(data_key='@dwell_times_line', default='1e-006 s')
    # that's important
    dwell_times_area = fields.Str(data_key='@dwell_times_area') 
    dwell_times_image = fields.Str(data_key='@dwell_times_image', default='1e-006 s')
    
    delay  = fields.Str(data_key='@delay', default='none')
    cycle_delay = fields.Str(data_key='@cycle_delay', default='0 s')

    dose_image = fields.Str(data_key='@dose_image', default='0 C/m&#178;')
    # that's important
    dose_area = fields.Str(data_key='@dose_area') 
    dose_line = fields.Str(data_key='@dose_line', default='0 C/m')
    dose_point = fields.Str(data_key='@dose_point', default='0 C')
    
    pause = LCBool(data_key='@pause', default=False)
    
    scanning_mode_fast = fields.Str(data_key='@scanning_mode_fast', default='bidirectional')
    scanning_mode_cycle_mode =  fields.Str(data_key='@scanning_mode_cycle_mode', default='back-and-forth')
    
    pixel_spacing_image = fields.Str(data_key='@pixel_spacing_image', default='0 m')
    # that's important
    pixel_spacing_area = fields.Str(data_key='@pixel_spacing_area')
    pixel_spacing_line = fields.Str(data_key='@pixel_spacing_line', default='50 %')
    # this too?
    track_spacing = fields.Str(data_key='@track_spacing')
    
    description = fields.Str(data_key='@description', default='')

    probe = fields.Nested(ElyProbeSchema, data_key='PROBE')
    gis = fields.Nested(ElyGISSchema, data_key='GIS')


class ElyPolylineSchema(OrderedSchema):

    line_width = fields.Int(data_key='@line_width', default=0)
    points = fields.Str(data_key='@points')
    polyline = LCBool(data_key='@polyline', default=True)

class ElyArcSchema(OrderedSchema):
    arc_start = fields.Str(data_key='@arc_start', default='inside')

    cx = fields.Float(data_key='@cx')
    cy = fields.Float(data_key='@cy')
    r1 = fields.Float(data_key='@r1')
    r2 = fields.Float(data_key='@r2')
    a1 = fields.Str(data_key='@a1')
    a2 = fields.Str(data_key='@a2')

    exposure = fields.Nested(ElyExposureSchema, data_key='EXPOSURE', required=True)


class ElyPointSchema(OrderedSchema):

    x = fields.Float(data_key='@x')
    y = fields.Float(data_key='@y')

    exposure = fields.Nested(ElyExposureSchema, data_key='EXPOSURE', required=True)


from custom_backends.smart_fib_ely.models import ElyArc, ElyPolyline, ElyPoint


class ElyLayerReference(OrderedSchema):
    frame_cx = fields.Str(data_key='@frame_cx', default='0')
    frame_cy = fields.Str(data_key='@frame_cy', default='0')
    fov = fields.Int(data_key='@frame_size')
    name = fields.Str(data_key='@ref')

    # polylines = fields.List(
    #     PolyFieldWithCustomDataKey(
    #     serialization_schema_selector=shape_schema_serialization_disambiguation,
    #     deserialization_schema_selector=None, # shape_schema_deserialization_disambiguation,
    #     required=True
    # ))

    @pre_dump
    def foo(self, obj, many):
        # print(self.__dict__)

        for i, geom in enumerate(obj.geoms):
            # print(type(geom))
            if isinstance(geom, ElyPoint):
                self.__dict__['dump_fields'][f'{i}'] = fields.Nested(
                    ElyPointSchema(), data_key=f'POINT#-#{i}#-#', default=geom, dump_only=True
                )
            elif isinstance(geom, ElyPolyline):
                self.__dict__['dump_fields'][f'{i}'] = fields.Nested(
                    ElyPolylineSchema, data_key=f'LINES#-#{i}#-#', default=geom, dump_only=True
                )
            elif isinstance(geom, ElyArc):
                self.__dict__['dump_fields'][f'{i}'] = fields.Nested(
                    ElyArcSchema, data_key=f'ARC#-#{i}#-#', default=geom, dump_only=True
                )
            else:
                RuntimeError(f'Cannot export {geom}.')

        return obj

    # polylines = fields.List(fields.Nested(ElyPolylineSchema), data_key='LINES')
    # points = fields.List(fields.Nested(ElyPointSchema), data_key='POINT')


class ElyStructureSchema(OrderedSchema):
    locked = LCBool(data_key='@locked', default=False)
    name = fields.Str(data_key='@name')

    version = fields.Nested(ElyVersionSchema, data_key='VERSION')

    instance_list = fields.Nested(ElyInstanceList, data_key='INSTANCE_LIST')

    layers = fields.List(fields.Nested(ElyLayerReference), data_key='LAYER_REFERENCE')


class ElyStructureListSchema(OrderedSchema):
    structures = fields.List(fields.Nested(ElyStructureSchema), data_key='STRUCTURE')


class ElyFileSchema(EnvelopeBaseSchema):
    __envelope__ = 'ELAYOUT'

    version_attr = fields.Str(data_key='@version', default="2.0")
    locked = LCBool(data_key='@locked', default=False)
    name = fields.Str(data_key='@name')

    version = fields.Nested(ElyVersionSchema, data_key='VERSION')
    axes = fields.Nested(ElyAxesSchema, data_key='AXES')
    grid = fields.Nested(ElyGridSchema, data_key='GRID')

    layer_list = fields.Nested(ElyLayerListSchema, data_key='LAYER_LIST')

    structure_list = fields.Nested(ElyStructureListSchema, data_key='STRUCTURE_LIST')
