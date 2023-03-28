import taichi as ti
from .utils import *

@ti.func
def Hash21(Vec2):
    Vec2 = fract(Vec2 * vec2(234.34, 435.345))
    Vec2 += dot(Vec2, Vec2 + 43.23)
    return fract(Vec2.x * Vec2.y)

@ti.func
def pulsations(t):
    """
    Heartbeat pulsations rule
    """
    return (ti.sin(t * 3.) + ti.cos(4.4 * t / 2.) + 4) / 6


@ti.func
def mainImage(fragCoord, iTime: ti.f32, iResolution):

    col = vec3(0)
    flow_color = vec3(222., 52., 10.) / 255.
    longwise_color = vec3(242., 255., 151.) / 255.
    central_color = vec3(78., 16., 105.) / 255.

    # narrowing shapes to frame borders
    absolute_uv = fragCoord.xy / iResolution.xy - 0.5
    thinning = abs(length(absolute_uv) - 1.)

    # center uv and norm y in [-1/2, 1/2] (x in [-1/2 w/h, 1/2 w/h])
    uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y 

    PI = 3.14159256

    '''Water Shaders in the corners'''
    '''Translate, rotate, translate back'''
    addition = vec2(-1. * iResolution.x / (2. * iResolution.y), -0.5)
    uv += addition
    uv = multiply2_left(rot(PI * pow((1.42 - pow(length(uv), .5)), 15.)), uv)
    uv -= addition
  
    '''Translate, rotate, translate back'''
    addition = vec2(iResolution.x / (2. * iResolution.y), 0.5)
    uv += addition
    uv =  multiply2_left(rot(PI * pow((1.42 - pow(length(uv), .5)), 15.)), uv)
    uv -= addition

    # general move
    uv += iTime * 0.04
    # objects size
    uv *= 6.

    '''water effect'''
    X = uv.x * 25. + iTime
    Y = uv.y * 25. + iTime
    uv.y += cos(X + Y) * 0.1 * cos(Y) * length(absolute_uv)
    uv.x += sin(X - Y) * 0.1 * sin(Y) * length(absolute_uv)

    # division of the area into squares (Truchet Tiling)
    grid_view = fract(uv) - 0.5
    id = floor(uv)

    # randomly turn pattern squares (tile)
    rand_n = Hash21(id) # random number between 0 and 5
    if rand_n < 0.5: grid_view.x *= -1.


    # define pulsations as thickening of figures
    width = 0.2 * thinning * pulsations(iTime)

    curve = grid_view - sign(grid_view.x + grid_view.y + 0.001) * 0.5
    d = length(curve)
    # the angle laid off from one of the corners of the tile
    angle = atan(curve.x, curve.y) # angle -pi, pi

    # -1 of 1 (if tile odd or even)
    checker = mod(id.x + id.y, 2.) * 2. - 1.
    
    # mask for the figures
    mask = smoothstep(0.01, -0.01, abs(d - 0.5) - width) # 0.5 - radius

    # rule for running stripes
    flow = fract(sin(iTime + checker * angle * 10.))

    central_canvas_gradient = thinning

    longwise_gradient = (d - (0.5 - width)) / (2. * width)
    longwise_gradient = abs(abs(longwise_gradient - 0.5) * 2. - 1.)

    final_gradient = flow_color * flow + central_color * central_canvas_gradient + longwise_color * longwise_gradient

    col += final_gradient * mask

    col *= pow(thinning, 2.)

    return col