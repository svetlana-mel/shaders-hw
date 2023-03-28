from shader_1 import run as run_shader_1
from shader_2 import run as run_shader_2

import sys

SHADER_NUMBER = 0

if __name__ == "__main__":
    try:
        SHADER_NUMBER = int(sys.argv[1])
        if SHADER_NUMBER not in set((1, 2)):
            print("Wrong shader number. Avaliable 1 or 2.")
            SHADER_NUMBER = 1
    except:
        SHADER_NUMBER = 1

    print(f'Shader {SHADER_NUMBER} is running.')

    run_shader_1() if SHADER_NUMBER == 1 else run_shader_2()
