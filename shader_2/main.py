import taichi as ti
from .utils import *
from .mainImage import mainImage
from .cracks import background
import time

def run():
    # Initializes the Taichi runtime.
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    # ti.init(debug=True)

    # resolution and pixels
    asp = 16/9
    h = 600
    w = int(asp * h)
    resolution = w, h
    iResolution = vec2(w, h)

    # 3-dimentional vector field
    pixels = ti.Vector.field(3, dtype=ti.f32, shape=resolution)

    @ti.kernel
    def render(iTime: ti.f32, frame: ti.int32):
        for fragCoord in ti.grouped(pixels):
            pixels[fragCoord] = mainImage(fragCoord, iTime, iResolution)
            # pixels[fragCoord] = background(fragCoord, iTime, iResolution)



    gui = ti.GUI("Shader #1", res=resolution, fast_gui=True)
    frame = 0
    start = time.time()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                break

        iTime = time.time() - start
        render(iTime, frame)
        gui.set_image(pixels)
        gui.show()
        frame += 1

    gui.close()

