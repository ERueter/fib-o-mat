import sys

import numpy as np

from vispy import app
from vispy.gloo import context
from vispy.visuals import TextVisual, RectangleVisual
from vispy.visuals.transforms import STTransform


# https://github.com/vispy/vispy/blob/a6ec64bb7b56739b463e3c16245a31940ca4f5f7/vispy/visuals/text/text.py#L151
def _glpyh_dimensions(text, font, anchor_x, anchor_y, lowres_size):
    """Convert text characters to VBO"""
    # Necessary to flush commands before requesting current viewport because
    # There may be a set_viewport command waiting in the queue.
    # TODO: would be nicer if each canvas just remembers and manages its own
    # viewport, rather than relying on the context for this.
    canvas = context.get_current_canvas()
    canvas.context.flush_commands()

    text_vtype = np.dtype([('a_position', np.float32, 2),
                           ('a_texcoord', np.float32, 2)])
    vertices = np.zeros(len(text) * 4, dtype=text_vtype)
    prev = None
    width = height = ascender = descender = 0
    ratio, slop = 1. / font.ratio, font.slop
    x_off = -slop
    # Need to make sure we have a unicode string here (Py2.7 mis-interprets
    # characters like "•" otherwise)
    if sys.version[0] == '2' and isinstance(text, str):
        text = text.decode('utf-8')
    # Need to store the original viewport, because the font[char] will
    # trigger SDF rendering, which changes our viewport
    # todo: get rid of call to glGetParameter!

    # Also analyse chars with large ascender and descender, otherwise the
    # vertical alignment can be very inconsistent
    for char in 'hy':
        glyph = font[char]
        y0 = glyph['offset'][1] * ratio + slop
        y1 = y0 - glyph['size'][1]
        ascender = max(ascender, y0 - slop)
        descender = min(descender, y1 + slop)
        height = max(height, glyph['size'][1] - 2*slop)

    # Get/set the fonts whitespace length and line height (size of this ok?)
    glyph = font[' ']
    spacewidth = glyph['advance'] * ratio
    lineheight = height * 1.5

    # Added escape sequences characters: {unicode:offset,...}
    #   ord('\a') = 7
    #   ord('\b') = 8
    #   ord('\f') = 12
    #   ord('\n') = 10  => linebreak
    #   ord('\r') = 13
    #   ord('\t') = 9   => tab, set equal 4 whitespaces?
    #   ord('\v') = 11  => vertical tab, set equal 4 linebreaks?
    # If text coordinate offset > 0 -> it applies to x-direction
    # If text coordinate offset < 0 -> it applies to y-direction
    esc_seq = {7: 0, 8: 0, 9: -4, 10: 1, 11: 4, 12: 0, 13: 0}

    # Keep track of y_offset to set lines at right position
    y_offset = 0

    # When a line break occur, record the vertices index value
    vi_marker = 0
    ii_offset = 0  # Offset since certain characters won't be drawn

    # The running tracker of characters vertex index
    vi = 0

    orig_viewport = canvas.context.get_viewport()
    for ii, char in enumerate(text):
        if ord(char) in esc_seq:
            if esc_seq[ord(char)] < 0:
                # Add offset in x-direction
                x_off += abs(esc_seq[ord(char)]) * spacewidth
                width += abs(esc_seq[ord(char)]) * spacewidth
            elif esc_seq[ord(char)] > 0:
                # Add offset in y-direction and reset things in x-direction
                dx = dy = 0
                if anchor_x == 'right':
                    dx = -width
                elif anchor_x == 'center':
                    dx = -width / 2.
                vertices['a_position'][vi_marker:vi+4] += (dx, dy)
                vi_marker = vi+4
                ii_offset -= 1
                # Reset variables that affects x-direction positioning
                x_off = -slop
                width = 0
                # Add offset in y-direction
                y_offset += esc_seq[ord(char)] * lineheight
        else:
            # For ordinary characters, normal procedure
            glyph = font[char]
            kerning = glyph['kerning'].get(prev, 0.) * ratio
            x0 = x_off + glyph['offset'][0] * ratio + kerning
            y0 = glyph['offset'][1] * ratio + slop - y_offset
            x1 = x0 + glyph['size'][0]
            y1 = y0 - glyph['size'][1]
            u0, v0, u1, v1 = glyph['texcoords']
            position = [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
            texcoords = [[u0, v0], [u0, v1], [u1, v1], [u1, v0]]
            vi = (ii + ii_offset) * 4
            vertices['a_position'][vi:vi+4] = position
            vertices['a_texcoord'][vi:vi+4] = texcoords
            x_move = glyph['advance'] * ratio + kerning
            x_off += x_move
            ascender = max(ascender, y0 - slop)
            descender = min(descender, y1 + slop)
            width += x_move
            height = max(height, glyph['size'][1] - 2*slop)
            prev = char

    if orig_viewport is not None:
        canvas.context.set_viewport(*orig_viewport)

    dx = dy = 0
    if anchor_y == 'top':
        dy = -descender
    elif anchor_y in ('center', 'middle'):
        dy = (-descender - ascender) / 2
    elif anchor_y == 'bottom':
        dy = -ascender
    if anchor_x == 'right':
        dx = -width
    elif anchor_x == 'center':
        dx = -width / 2.

    # If any linebreaks occured in text, we only want to translate characters
    # in the last line in text (those after the vi_marker)
    vertices['a_position'][0:vi_marker] += (0, dy)
    vertices['a_position'][vi_marker:] += (dx, dy)
    vertices['a_position'] /= lowres_size

    return width, height


class ScaleBar:
    def __init__(self, unit: str, parent: app.Canvas):
        self._unit = unit

        self._parent = parent

        self._background = RectangleVisual(center=(0, 0), width=20, height=5, color=(1, 1, 1, .75))
        self._background.transform = STTransform(
            translate=(self._parent.physical_size[0] / 2., self._parent.physical_size[1] / 2.)
        )

        self._scale = RectangleVisual(center=(0, 0), width=20, height=5, color=(0,0,0,1))
        self._scale.set_gl_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self._scale.transform = STTransform(
            translate=(self._parent.physical_size[0] / 2., self._parent.physical_size[1] / 2.)
        )

        self._label = TextVisual('10 µm', bold=False, font_size=12, color='black', pos=(0, 10), anchor_x='center', anchor_y='bottom')
        self._label.transform = STTransform(
            translate=(self._parent.physical_size[0] / 2., self._parent.physical_size[1] / 2.)
        )

    def transform(self, pixel_per_length, scale):
        vp = (0, 0, self._parent.physical_size[0], self._parent.physical_size[1])

        diag = np.sqrt(self._parent.physical_size[0]**2 + self._parent.physical_size[1]**2)

        self._scale.transforms.configure(canvas=self._parent, viewport=vp)
        self._scale.transform.translate=(self._parent.physical_size[0] / 2, self._parent.physical_size[1] / 2)
        self._scale.transform.scale = scale, scale

        self._scale.height = 10 / scale

        scale_bar_length = pixel_per_length * self._parent.physical_size[0] / 5
        if scale_bar_length > 1:
            new_rect_width = 10**int(np.log10(scale_bar_length)) / pixel_per_length / scale
        else:
            new_rect_width = 10**int(np.log10(scale_bar_length) - 1) / pixel_per_length / scale

        self._scale.width = 1 * new_rect_width

        self._label.transforms.configure(canvas=self._parent, viewport=vp)
        self._label.transform.translate=(self._parent.physical_size[0] / 2, self._parent.physical_size[1] / 2)
        self._label.text = f'{pixel_per_length * self._scale.width * scale:.2f} {self._unit}'

        glyph_dims = np.array([
            _glpyh_dimensions(
                t, self._label._font, self._label._anchors[0], self._label._anchors[1], self._label._font._lowres_size
            ) for t in self._label.text
        ])

        self._background.transforms.configure(canvas=self._parent, viewport=vp)
        self._background.transform.translate=(self._parent.physical_size[0] / 2, self._parent.physical_size[1] / 2 + 15)
        self._background.transform.scale = scale, scale

        if self._scale.width > np.sum(glyph_dims[:, 0]) / scale / 2:
            self._background.width = 1.1 * self._scale.width
        else:
            self._background.width = 1.5 * np.sum(glyph_dims[:, 0]) / scale / 2

        self._background.height = 80 / scale

        margin = 10 * scale

        shift_background = (
            self._parent.physical_size[0] - scale * self._background.width / 2 - margin,
            self._parent.physical_size[1] - scale * self._background.height / 2 - margin
        )
        shift = (shift_background[0], shift_background[1] - 15)

        self._background.transform.translate = shift_background
        self._label.transform.translate = shift
        self._scale.transform.translate = shift

    def draw(self):
        self._background.draw()
        self._scale.draw()
        self._label.draw()
