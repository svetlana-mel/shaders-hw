import taichi as ti

vec2 = ti.types.vector(2, ti.f32)
vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)

@ti.func
def fract(x):
    """
    compute the fractional part of the argument
    x : scalar or vector
    """
    return x - ti.floor(x, ti.f32)


@ti.func
def clamp(x, minVal, maxVal):
    """
    constrain a value to lie between two further values
    Parameters:
    x : Specify the value to constrain.

    minVal : Specify the lower end of the range into which to constrain x.

    maxVal : Specify the upper end of the range into which to constrain x.
    """
    return ti.min(ti.max(x, minVal), maxVal)


@ti.func
def smoothstep(edge0, edge1, x):
    """
    performs smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1

    Parameters:
    edge0 : Specifies the value of the lower edge of the Hermite function.

    edge1 : Specifies the value of the upper edge of the Hermite function.

    x : Specifies the source value for interpolation.
    """
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@ti.func
def floor(x):
    """
    returns a value equal to the nearest integer that is less than or equal to x.
    """
    return ti.floor(x, ti.f32)


@ti.func
def dot(p, q):
    return p.dot(q)

@ti.func
def length(vec):
    """ vector or scalar length """
    return ti.sqrt(vec.dot(vec))

@ti.func
def sign(x: ti.f32):
    return 1. if x > 0. else -1. if x < 0. else 0.

@ti.func
def atan(x: ti.f32, y: ti.f32):
    return ti.atan2(x, y)

@ti.func
def asin(x: ti.f32):
    return ti.asin(x)

@ti.func
def sin(x: ti.f32):
    return ti.sin(x)

@ti.func
def cos(x: ti.f32):
    return ti.cos(x)

@ti.func
def mod(x: ti.f32, y: ti.f32):
    return x - y * floor(x/y)

@ti.func
def sqrt(vec):
    return ti.sqrt(vec)
