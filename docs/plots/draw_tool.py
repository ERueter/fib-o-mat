""" Demonstration of how to register event callbacks using an adaptation
of the color_scatter example from the bokeh gallery
"""
import numpy as np



from bokeh import events
from bokeh.io import output_file, show
from bokeh.layouts import column, row
from bokeh.models import Button, CustomJS, Div, Label, CrosshairTool, Tool, Inspection
from bokeh.plotting import figure
from bokeh.util.compiler import TypeScript
from bokeh.core import properties as bokeh_prop



TS_CODE = """
import {EditTool, EditToolView} from "models/tools/edit/edit_tool"

import {Renderer} from "models/renderers/renderer"
import {Label} from "models/annotations/label"
import {PanEvent, MoveEvent, TapEvent} from "core/ui_events"
import * as p from "core/properties"
import {Color} from "core/types"
import {values} from "core/util/object"
import {bk_tool_icon_range} from "styles/icons"

import {Annotation, AnnotationView} from "models/annotations/annotation"
import * as mixins from "core/property_mixins"
import {Line} from "core/visuals"

export class LineMeasureView extends AnnotationView {
    model: LineMeasure
    visuals: LineMeasure.Visuals

    connect_signals(): void {
        super.connect_signals()
        this.connect(this.model.change, () => this.plot_view.request_paint(this))
    }

    render(): void {
        if (!this.model.visible)
            return

        const {ctx} = this.layer
        ctx.save()

        ctx.beginPath()
        this.visuals.line.set_value(ctx)
        ctx.moveTo(this.model.x_start, this.model.y_start)
        ctx.lineTo(this.model.x_end, this.model.y_end)

        ctx.stroke()

        ctx.restore()
    }
}

export namespace LineMeasure {
    export type Attrs = p.AttrsOf<Props>

    export type Props = Annotation.Props & {
        x_start: p.Property<number>
        y_start: p.Property<number>
        x_end: p.Property<number>
        y_end: p.Property<number>
    } & Mixins

    export type Mixins = mixins.Line/*Scalar*/

    export type Visuals = Annotation.Visuals & { line: Line }
}

export interface LineMeasure extends LineMeasure.Attrs {
}

export class LineMeasure extends Annotation {
    properties: LineMeasure.Props
    __view_type__: LineMeasureView

    constructor(attrs?: Partial<LineMeasure.Attrs>) {
        super(attrs)
    }

    static init_LineMeasure(): void {
        this.prototype.default_view = LineMeasureView

        this.mixins<LineMeasure.Mixins>(mixins.Line/*Scalar*/)

        this.define<LineMeasure.Props>({
            x_start: [p.Number],
            x_end: [p.Number],
            y_start: [p.Number],
            y_end: [p.Number],
        })

        /*
        this.override({
          line_color: 'black',
        })
        */
    }
}

export class MeasureToolView extends EditToolView {
    model: MeasureTool

    connect_signals(): void {
        super.connect_signals()
        this.connect(this.model.properties.active.change, () => this._active_change())
    }

    _active_change(): void {
        if (!this.model.active)
            this._reset_measure_band()
    }

    _reset_measure_band(): void {
        this.model.measure_band.scale.x_start = 0
        this.model.measure_band.scale.y_start = 0

        this.model.measure_band.scale.x_end = 0
        this.model.measure_band.scale.y_end = 0

        this.model.measure_band.label.text = ""
        this.model.measure_band.pos_label.text = ""
    }

    /*
    _scroll(_ev: ScrollEvent): void {
        // const {delta} = _ev
        // if (delta != 0)
        this._reset_measure_band()
    }
     */

    _move(_ev: MoveEvent): void {
        const {sx, sy} = _ev
        const {frame} = this.plot_view

        const x = frame.xscales.default.invert(sx)
        const y = frame.yscales.default.invert(sy)

        this.model.measure_band.pos_label.text =
             "x=" + Number(x).toFixed(2) + " " + this.model.measure_unit +
             ", y=" + Number(y).toFixed(2) + " " + this.model.measure_unit

    }

    _move_exit(_e: MoveEvent): void {
        this.model.measure_band.pos_label.text = ""
    }

    _tap(_e: TapEvent): void {
        this._reset_measure_band()
    }

    _pan_start(_ev: PanEvent): void {
        const {sx, sy} = _ev

        this.model.measure_band.scale.x_start = sx
        this.model.measure_band.scale.y_start = sy
    }

    _pan(_ev: PanEvent): void {
        if (!this.model.active)
            return

        const {sx, sy} = _ev
        const {frame} = this.plot_view

        this.model.measure_band.scale.x_end = sx
        this.model.measure_band.scale.y_end = sy

        const x_end = frame.xscales.default.invert(sx)
        const y_end = frame.yscales.default.invert(sy)

        const x_start = frame.xscales.default.invert(this.model.measure_band.scale.x_start)
        const y_start = frame.yscales.default.invert(this.model.measure_band.scale.y_start)

        const dx = x_end - x_start
        const dy = y_end - y_start
        const distance = Math.sqrt(dx * dx + dy * dy)

        const angle = Math.atan2(dy, dx) * 180 / Math.PI

        this.model.measure_band.label.text =
             "distance=" + Number(distance).toFixed(2) + " " + this.model.measure_unit +
             ", angle= " + Number(angle).toFixed(2) + "°"
    }
}

export namespace MeasureTool {
    export type Attrs = p.AttrsOf<Props>

    export type Props = EditTool.Props & {
        measure_unit: p.Property<string>

        line_dash: p.Property<number[]>
        line_color: p.Property<Color>
        line_width: p.Property<number>
        line_alpha: p.Property<number>

        measure_band: p.Property<{ scale: LineMeasure, label: Label, pos_label: Label }>
    }
}

export interface MeasureTool extends MeasureTool.Attrs {
}

export class MeasureTool extends EditTool {
    properties: MeasureTool.Props
    __view_type__: MeasureToolView

    constructor(attrs?: Partial<MeasureTool.Attrs>) {
        super(attrs)
    }

    static init_MeasureTool(): void {
        this.prototype.default_view = MeasureToolView

        this.define<MeasureTool.Props>({
            measure_unit: [p.String],
            line_dash: [p.Array, [4, 4]],
            line_color: [p.Color, 'black'],
            line_width: [p.Number, 1.5],
            line_alpha: [p.Number, 1],
        })

        this.internal({
            measure_band: [p.Any],
        })

        this.register_alias("measure", () => new MeasureTool())
    }

    tool_name = "Measure"
    icon = bk_tool_icon_range
    event_type = ["tap" as "tap", "pan" as "pan", "move" as "move", "scroll" as "scroll"]

    get synthetic_renderers(): Renderer[] {
        return values(this.measure_band)
    }

    initialize(): void {
        super.initialize()

        this.measure_band = {
            scale: new LineMeasure({
                x_start: 0,
                y_start: 0,
                x_end: 0,
                y_end: 0,
                line_dash: this.line_dash,
                line_color: this.line_color,
                line_width: this.line_width,
                line_alpha: this.line_alpha,
            }),
            label: new Label({
                x: 10,
                y: 35,
                x_units: "screen",
                y_units: "screen",
                text: "",
                text_font_size: "18px",
                background_fill_alpha: .75,
                background_fill_color: "white"
            }),
            pos_label: new Label({
                x: 10,
                y: 10,
                x_units: "screen",
                y_units: "screen",
                text: "",
                text_font_size: "18px",
                background_fill_alpha: .75,
                background_fill_color: "white"
            }),
        }
    }
}

"""


class MeasureTool(Inspection):
    """
    A measure tool for bokeh plots!
    """

    __implementation__ = TypeScript(code=TS_CODE)

    measure_unit = bokeh_prop.String(default='apples', help='')

    line_dash = bokeh_prop.DashPattern(default='solid', help='')

    line_color = bokeh_prop.Color(default="black", help='')

    line_width = bokeh_prop.Float(default=1, help='')

    line_alpha = bokeh_prop.Float(default=1.0, help='')

def display_event(label, unit):
    "Build a suitable CustomJS to display the current event in the div model."


x = np.random.random(size=4000)*100
y = np.random.random(size=4000)*100
radii = np.random.random(size=4000)*1.5
colors = ["#%02x%02x%02x"%(int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

p = figure(tools='pan,wheel_zoom')
p.scatter(x, y, radius=np.random.random(size=4000)*1.5,
          fill_color=colors, fill_alpha=0.6, line_color=None)

p.add_tools(
    MeasureTool(
        measure_unit=f'bananas',  line_width=3
    )
)

# coord_label = Label(
#     text="",
#     x=10, y=10,
#     x_units='screen', y_units='screen',
#     text_font_size='20px',
#     background_fill_alpha=.5, background_fill_color='white'
# )
# p.add_layout(coord_label)
#
# p.js_on_event(
#     events.MouseMove,
#     CustomJS(
#         args=dict(label=coord_label, unit='µm'),
#         code="""
#             label.text = "x=" + Number(cb_obj["x"]).toFixed(2) + " y=" + Number(cb_obj["x"]).toFixed(2) + " " + unit;
#         """
#     )
# )
#
# p.js_on_event(
#     events.MouseLeave,
#     CustomJS(
#         args=dict(label=coord_label, unit='µm'),
#         code="""label.text = "";"""
#     )
# )

# output_file("js_events.html", title="JS Events Example")
show(p)
