import taichi as ti
from .utils import *
from .cracks import background


@ti.func
def rand(x: ti.f32):
    return fract(sin(x) * 43758.5453)


@ti.func
def cbrt(f):
    '''utility function to calculate the 3rd root. The pow() function has some problems with signs.'''
    return sign(f) * pow(abs(f), 1.0 / 3.0)

@ti.func
def hypot(a, b):
    '''distance between two points'''
    return length(vec2(a, b))


@ti.func
def bezier(a, b, c, p):
    '''
    Returns the exact distance to a quadratic bezier curve.

    a, c : start and end points of the curve
    b : control point of the curve
    p : current pixel coordinates
    '''
    ny          = vec2(normalize(a - 2.0 * b + c))
    nx          = vec2(ny.y, -ny.x)
    xa: ti.f32  = dot(a - b, ny) / dot(a - b, nx) / 2.0
    xc: ti.f32  = dot(c - b, ny) / dot(c - b, nx) / 2.0
    scale: ti.f32 = (xa - xc) / dot(a - c, nx)
    vertex      = a - nx * (xa / scale) - ny * (xa * xa / scale)

    px: ti.f32    = dot(p - vertex, nx) * scale
    py: ti.f32    = dot(p - vertex, ny) * scale
    min_x: ti.f32 = min(xa, xc)
    max_x: ti.f32 = max(xa, xc)
    '''
    // (px,py) are transformed such that we just need to find their distance to
    // the parabola y=x^2.
    // For that we have to find the closest point q on the parabola. Note that
    // q = (qx, qy) and qy = qx^2.
    // A perpendicular line through the parabola at q should intersect (px,py).
    // The function of a line normal to the parabola is y=0.5-x/(2qx)+qy.
    // Plugging in px,py we get
    // py = 0.5-px/(2qx)+qy = 0.5-px/(2qx)+qx^2
    // py+px/(2qx) = 0.5+qx^2
    // qx*py+px/2 = qx/2+qx^3
    // qx^3 + (1/2-py)*qx - px/2 = 0
    // Finding the roots of this equation tells us the x-coordinate of the
    // closest point. Squaring that will also give us the y-coordinate.
    // The nice thing is, that this cubic equation is already depressed, which
    // means it has no quadratic component and the cubic has a scalar of 1.
    '''
    l = 0.5 - py
    
    e = -(l * l * l / 27.0)
    dis = px * px * 0.25 - 4.0 * e

    result = 0.

    if (0.0 <= dis):
        '''one root'''
        f = px * 0.25 + sign(px) * sqrt(dis) * 0.5
        qx = clamp(cbrt(f) + cbrt(e / f), min_x, max_x)
        result = hypot(qx - px, qx * qx - py) / scale
    else:
        '''three roots
        However, the center one can never be the closest, so we can ignore it.'''
        r3p = sqrt(py - 0.5) * (2.0 / sqrt(3.0))
        ac = acos(-1.5 * px / (l * r3p)) / 3.0
        qx0 = clamp(r3p * cos(ac              ), min_x, max_x)
        qx1 = clamp(r3p * cos(ac - 4.188790205), min_x, max_x)
    
        result =  min(
            hypot(qx0 - px, qx0 * qx0 - py),
            hypot(qx1 - px, qx1 * qx1 - py)) / scale
    return result

@ti.func
def segment(a, b, p):
    ba = vec2(b - a)
    pa = vec2(p - a)
    h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)
    return length(pa - h * ba)

@ti.func
def high_between(f, lo, hi, iResolution):
    d = 2.0 / iResolution.x
    rad = (hi - lo) / 2.0
    mid = (lo + hi) / 2.0
    return smoothstep(-d, d, rad - abs(f - mid))

@ti.func
def pattern(i: int, p, iResolution):
    i = i % 4
    s = (p.x - p.y) / sqrt(2.0)

    result = 0.
    if (0 == i):
        result = high_between(mod(s, 0.03), 0.2 * 0.03, 0.55 * 0.03, iResolution)
    
    if (1 == i):
        m = mod(s, 0.03)
        result = high_between(m, 0.1 * 0.03, 0.3 * 0.03, iResolution) + high_between(m, 0.5 * 0.03, 0.8 * 0.03, iResolution)
    
    if (2 == i):
        result = high_between(mod(s, 0.01), 0.2 * 0.01, 0.65 * 0.01, iResolution)
    
    if (3 == i):
        rot = sqrt(2.0) / 2.0 * mat2( 1.0, -1.0, 1.0,  1.0)
        dot_center = multiply2_left(transpose(rot), round(multiply2_left(rot, p) * 100.0)) / 100.0
        dot_radius = mix(rand(dot_center.x + dot_center.y), 1.0, 0.8) * 0.003
        result = high_between(length(dot_center - p), dot_radius, 100.0, iResolution)

    return result




@ti.func
def mainImage(fragCoord, iTime: ti.f32, iResolution):
    ''' CONSTANTS '''
    red    = vec3(0.816, 0.325, 0.227)
    green  = vec3(0.584, 0.639, 0.38)
    blue   = vec3(0.498, 0.588, 0.49)
    yellow = vec3(0.843, 0.725, 0.353)
    white  = vec3(0.91,  0.804, 0.596)
    black  = vec3(0.125, 0.098, 0.078)

    NUM_BELTS = 5

    fg_colors = mat83(blue,    red,  green, green, yellow,  blue,   red, green)
    bg_colors = mat83(red, yellow, yellow,  blue,  white, white, white, white)


    ''' (0,0) at the center, -1 left, 1 right, -1 bottom, 1 top. '''
    p = (2.0 * fragCoord.xy - iResolution.xy) / iResolution.x

    '''
    add two levels of noise to the pixel position:
    1. some coarse noise to make the likes look more hand-drawn.
    '''
    p += vec2(sin(p.x * 64.0 + p.y * 128.0) * 0.000625, 
              sin(p.y * 64.0 + p.x *  32.0) * 0.000625)
    '''2. some fine noise to make the edges look more like ink on paper. '''
    p += vec2(rand(p.x * 31.0 + p.y * 87.0) * 0.001,
                rand(p.x * 11.0 + p.y * 67.0) * 0.001)
    
    outline = 0.0
    id = -1.0
    for i in range(0, NUM_BELTS):
        t = iTime + 16. * i + 1024.0
        p0 =      vec2(-1.5,              sin(t * 0.02))
        p1 =      vec2( sin(t*0.1) * 0.1, sin(t * 0.07) * 0.7)
        p2 =      vec2( 1.5,              sin(t * 0.03))
        c0 = p1 + vec2(-0.5,              sin(t * 0.13) * 0.5)
        c1 = 2.0 * p1 - c0

        dist = min(
            bezier(
                p0 if p.x < p1.x else p1,
                c0 if p.x < p1.x else c1,
                p1 if p.x < p1.x else p2,
                p),
            segment(
                p1,
                c1 if p.x < p1.x else c0,
                p)
            )
        p3 = p0 - vec2(1.0 - sin(t * 0.025), 1.0 - sin(t * 0.027)) * 0.05
        p4 = p1 + vec2(      sin(t * 0.014),       sin(t * 0.032)) * 0.05
        p5 = p2 + vec2(1.0 - sin(t * 0.014), 1.0 - sin(t * 0.032)) * 0.05
        c2 = p4 + vec2(-1.0,                       sin(t * 0.13))  * 0.5
        c3 = 2.0 * p4 - c0
        dist2 = min(
            bezier(
                p3 if p.x < p4.x else p4,
                c2 if p.x < p4.x else c3,
                p4 if p.x < p4.x else p5,
                p),
            segment(
                p4,
                c3 if p.x < p4.x else c2,
                p)
            )
        dist = min(dist, dist2 + 0.01)

        dist *= sin(p.x * 10.0 + sin(p.y)) * 0.2 + 1.0

        fill   = high_between(dist, -1.0,   0.025, iResolution)
        border = high_between(dist,  0.022, 0.028, iResolution)
        id      = mix(id, ti.cast(i, ti.f32), fill)
        outline = mix(outline, border, fill)

    background_color = background(fragCoord, iTime, iResolution)

    fg = fg_colors[ti.cast(id / 4, ti.i32), :] if 0.0 <= id else background_color
    bg = bg_colors[ti.cast(id / 4, ti.i32), :] if 0.0 <= id else background_color
    color = vec3(0.)
    color = mix(
        mix(fg, 
            bg, 
            pattern(int(id), p, iResolution)
        ), 
        black, 
        outline
    )

    '''Some noise to make it look more paper-y'''
    color *= 0.95 + rand(p.x + p.y) * 0.1

    return color