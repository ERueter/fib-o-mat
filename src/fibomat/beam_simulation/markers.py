import numpy as np

from vispy.visuals import Visual
from vispy.gloo import VertexBuffer


class Tail(Visual):
    VERTEX_SHADER = """
        #version 120
        uniform float u_size;

        attribute vec2 a_position;
        attribute vec4 a_color;

        varying vec4 v_fg_color;
        varying vec4 v_bg_color;
        varying float v_radius;
        varying float v_linewidth;
        varying float v_antialias;
        void main (void) {
            v_radius = u_size;
            v_linewidth = 1.0;
            v_antialias = 1.0;
            v_fg_color  = vec4(0.0,0.0,0.0,0.1);
            v_bg_color  = a_color;
            gl_Position = $transform(vec4(a_position,0,1));
            gl_PointSize = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
        }
    """

    FRAGMENT_SHADER = """
        #version 120
        varying vec4 v_fg_color;
        varying vec4 v_bg_color;
        varying float v_radius;
        varying float v_linewidth;
        varying float v_antialias;
        void main()
        {
            float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
            float t = v_linewidth/2.0-v_antialias;
            float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
            float d = abs(r - v_radius) - t;
            if( d < 0.0 )
                gl_FragColor = v_fg_color;
            else
            {
                float alpha = d/v_antialias;
                alpha = exp(-alpha*alpha);
                if (r > v_radius)
                    gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
                else
                    gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
            }
        }
    """

    def __init__(self, dwell_points: np.ndarray, tail_length: int, size: float):
        super().__init__(self.VERTEX_SHADER, self.FRAGMENT_SHADER)

        self.set_gl_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self._draw_mode = 'points'

        self._tail_length = tail_length
        self._size = size
        self._alpha = 0.5

        self._dwell_points = dwell_points

    def _prepare_transforms(self, view=None):
        view.view_program.vert['transform'] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        # print('prepare')
        # attributes / uniforms are not available until program is built
        pass

    def draw(self, current_point: int):
        if current_point > self._tail_length:
            i_min = current_point - self._tail_length
            i_max = current_point
        else:
            i_min = 0
            i_max = current_point

        n = i_max - i_min

        v_color = np.c_[np.full(n, 1.), np.zeros(n), np.zeros(n), np.linspace(0, self._alpha, n)].astype(np.float32)

        self.shared_program['a_position'] = VertexBuffer(self._dwell_points[i_min:i_max, :2])
        self.shared_program['a_color'] = VertexBuffer(v_color)
        self.shared_program['u_size'] = self._size

        super().draw()


class DwellPoints(Visual):
    VERTEX_SHADER = """
        #version 120
        uniform float u_size;
        uniform vec4 u_color;
        uniform float u_alpha_border;

        attribute vec2 a_position;

        varying vec4 v_fg_color;
        varying vec4 v_bg_color;
        varying float v_radius;
        varying float v_linewidth;
        varying float v_antialias;
        void main (void) {
            v_radius = u_size;
            v_linewidth = 1.0;
            v_antialias = 1.0;
            v_fg_color  = vec4(0.0,0.0,0.0,u_alpha_border);
            v_bg_color  = u_color;
            gl_Position = $transform(vec4(a_position,0,1));
            gl_PointSize = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
        }
    """

    FRAGMENT_SHADER = """
        #version 120
        varying vec4 v_fg_color;
        varying vec4 v_bg_color;
        varying float v_radius;
        varying float v_linewidth;
        varying float v_antialias;
        void main()
        {
            float size = 2.0*(v_radius + v_linewidth + 1.5*v_antialias);
            float t = v_linewidth/2.0-v_antialias;
            float r = length((gl_PointCoord.xy - vec2(0.5,0.5))*size);
            float d = abs(r - v_radius) - t;
            if( d < 0.0 )
                gl_FragColor = v_fg_color;
            else
            {
                float alpha = d/v_antialias;
                alpha = exp(-alpha*alpha);
                if (r > v_radius)
                    gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
                else
                    gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
            }
        }
    """

    def __init__(self, dwell_points: np.ndarray):
        super().__init__(self.VERTEX_SHADER, self.FRAGMENT_SHADER)
        self.set_gl_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
        self._draw_mode = 'points'

        self._dwell_points = dwell_points

        self.shared_program['a_position'] = VertexBuffer(dwell_points)
        self.shared_program['u_color'] = (0, 0, 0, 1)
        self.shared_program['u_size'] = 1
        self.shared_program['u_alpha_border'] = 1

    def _prepare_transforms(self, view=None):
        view.view_program.vert['transform'] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        pass
