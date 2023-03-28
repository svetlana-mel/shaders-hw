import taichi as ti
import math

# %% type shortcuts

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)

mat2 = ti.types.matrix(2, 2, ti.f32)

tmpl = ti.template()
# %% constants

twopi = 2 * math.pi
pi180 = math.pi / 180.

# %% shader language functions


@ti.func
def length(p):
    return ti.sqrt(p.dot(p))


@ti.func
def normalize(p):
    n = p.norm()
    return p / (n if n != 0. else 1.)


@ti.func
def mix(x, y, a):
    return x * (1. - a) + y * a


@ti.func
def dot(p, q):
    return p.dot(q)


@ti.func
def dot2(p):
    return p.dot(p)


@ti.func
def deg2rad(a):
    return a * pi180


@ti.func
def rot(a):
    c = ti.cos(a)
    s = ti.sin(a)
    return mat2([[c, -s], [s, c]])


@ti.func
def sign(x: ti.f32):
    return 1. if x > 0. else -1. if x < 0. else 0.


@ti.func
def signv(x: tmpl):
    r = ti.Vector(x.shape[0], x.dtype)
    for i in ti.static(range(x.shape[0])):
        r[i] = sign(x[i])
    return r


@ti.func
def sd_circle(p, r):
    return p.norm() - r


@ti.func
def sd_segment(p, a, b):
    pa = p - a
    ba = b - a
    h = clamp((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return (pa - ba * h).norm()


@ti.func
def sd_box(p, b):
    d = abs(p) - b
    return max(d, 0.).norm() + min(max(d.x, d.y), 0.0)


@ti.func
def sd_roundbox(p, b, r):
    rr = vec2(r[0], r[1]) if p[0] > 0. else vec2(r[2], r[3])
    rr[0] = rr[0] if p.y > 0. else rr[1]
    q = abs(p) - b + rr[0]
    return min(max(q[0], q[1]), 0.) + max(q, 0.0).norm() - rr[0]


@ti.func
def sd_trapezoid(p, r1, r2, he):
    k1 = vec2(r2, he)
    k2 = vec2(r2 - r1, 2. * he)
    pp = vec2(abs(p[0]), p[1])
    ca = vec2(pp[0] - min(pp[0], r1 if pp[1] < 0. else r2), abs(pp[1]) - he)
    cb = pp - k1 + k2 * clamp(dot(k1 - pp, k2) / dot2(k2), 0., 1.)
    s = -1. if cb[0] < 0. and ca[1] < 0. else 1.
    return s * ti.sqrt(min(dot2(ca), dot2(cb)))


@ti.func
def clamp(x, low, high):
    return ti.max(ti.min(x, high), low)


@ti.func
def fract(x):
    return x - ti.floor(x)


@ti.func
def step(edge, x):
    return 0. if x < edge else 1.


@ti.func
def smoothstep(edge0, edge1, x):
    n = (x - edge0) / (edge1 - edge0)
    t = clamp(n, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@ti.func
def smoothmin(a, b, k):
    h = ti.max(k - abs(a - b), 0.) / k
    return ti.min(a, b) - h * h * k * (1./4.)


@ti.func
def smoothmax(a, b, k):
    return smoothmin(a, b, -k)


@ti.func
def smoothmin3(a, b, k):
    h = ti.max(k - abs(a - b), 0.) / k
    return ti.min(a, b) - h * h * h * k * (1./6.)


@ti.func
def skewsin(x, t):
    return ti.atan2(t * ti.sin(x), (1. - t * ti.cos(x))) / t


@ti.func
def random2():
    return vec2(ti.random(ti.f32), ti.random(ti.f32))


@ti.func
def hash1(n):
    return fract(ti.sin(n) * 43758.5453)


@ti.func
def hash21(p):
    q = fract(p * vec2(123.34, 345.56))
    q += dot(q, q + 34.23)
    return fract(q.x * q.y)


@ti.func
def hash22(p):
    x = hash21(p)
    y = hash21(p + x)
    return vec2(x, y)


# https://www.shadertoy.com/view/ll2GD3
@ti.func
def pal(t, a, b, c, d):
    return a + b * ti.cos(twopi * (c * t + d))