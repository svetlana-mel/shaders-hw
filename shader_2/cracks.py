import taichi as ti
from .utils import *

@ti.func
def hash22(p):
    return fract(18.5453 * sin(multiply2_left(mat2(127.1, 311.7, 269.5, 183.3), p)))

@ti.func
def disp(p):
    ofs = .5; # jitter Voronoi centers in -ofs ... 1.+ofs
    return -ofs + (1. + 2. * ofs) * hash22(p)

@ti.func
def voronoiB(u): # returns len + id
    ''' 
    Voronoi distance to borders.
    inspired by https://www.shadertoy.com/view/ldl3W8
    return vec3
    u : vec2 
    '''
    iu = vec2(floor(u))
    C = vec2(0.)
    P = vec2(0.)
    d = 0.0
    m = 1e9

    for k in range(25):
        p = iu + vec2(ti.cast(k % 5 - 2, ti.f32), ti.cast(k // 5 - 2, ti.f32))
        o = disp(p)
        r = vec2(p - u + o)
        d = dot(r, r)
        if d < m: 
            m = d
            C = p - iu
            P = r

    m = 1e9
    
    for k in range(25):
        p = iu + C + vec2(ti.cast(k % 5 - 2, ti.f32), ti.cast(k // 5 - 2, ti.f32))
        o = disp(p)
        r = p - u + o

        if dot(P - r, P - r) > 1e-5:
            m = min(m, 0.5 * dot((P + r), normalize(r - P)))
    return vec3(m, P + u)

@ti.func
def hash21(p):
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123)

@ti.func
def noise2(p):
    i = vec2(floor(p)) # vec2
    f = fract(p) # vec2
    f = f * f * (3. - 2. * f); # smoothstep

    mix1 = mix(hash21(i + vec2(0., 0.)), 
                hash21(i + vec2(1., 0.)),
                f.x)

    mix2 = mix(hash21(i + vec2(0., 1.)),
                hash21(i + vec2(1., 1.)),
                f.x)
    v = mix(mix1, mix2, f.y)
    return 2. * v - 1.


@ti.func
def noise22(p):
    return vec2(noise2(p), noise2(p + 17.7))

@ti.func
def fbm22(p):
    '''
    add pseudo Perlin noise
    Parameter
    p : 2-d vector

    Return 
    v : 2-d vector with noise
    '''
    v = vec2(0.)
    a = .5
    R = rot(0.37)

    for _ in range(9): 
        p = multiply2_left(R, p)
        v += a * noise22(p)
        p *= 2.
        a /= 2.
    return v


@ti.func
def background(fragCoord, iTime: ti.f32, iResolution):
    U = vec2(ti.cast(fragCoord.x, ti.f32), ti.cast(fragCoord.y, ti.f32))
    U *= 5. / iResolution.y

    U.x += iTime / 8.
    U.y += (sin(iTime) / 20.) + 1.4

    RATIO = 2.

    CRACK_zebra_scale = .08
    CRACK_zebra_amp = (sin(iTime) / 20.) + 1.4
    CRACK_profile = 0.25 
    CRACK_slope = 1.4
    CRACK_width = .0


    V = U / vec2(RATIO, 1.) # voronoi cell shape
    ''' add pseudo Perlin noise '''
    D = fbm22(CRACK_zebra_scale * U) / CRACK_zebra_scale / CRACK_zebra_amp
    '''evaluate Voronoi distance to borders'''
    H = voronoiB(V + D); 
        
    d = H.x # distance to cracks

    d = min(1., CRACK_slope * pow(max(0., d - CRACK_width), CRACK_profile))

    white  = vec3(245.,  188., 126.) / 255.
  
    col = (sin(vec3(d)) + 0.3) * white

    return col
