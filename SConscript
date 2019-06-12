import copy
import hashlib
import itertools
import os

Import('env')

env.Object(
    'acf.o',
    'acf.cpp',
)

env.Test(
    'alias_test',
    [
        'alias_test.cpp',
    ],
)

env.Object(
    'config.o',
    'config.cpp',
)

env.Object(
    'flymc.o',
    'flymc.cpp',
)

env.Object(
    'fmh.o',
    'fmh.cpp',
)

env.Object(
    'io.o',
    'io.cpp',
)

env.Test(
    'lr_target_test',
    [
        'lr_target_test.cpp',
        'target_test.o',
    ],
)

env.Object(
    'lr_zigzag.o',
    'lr_zigzag.cpp',
)

env.Object(
    'mh.o',
    'mh.cpp',
)

env.Test(
    'rlr_target_test',
    [
        'rlr_target_test.cpp',
        'target_test.o',
    ],
)

env.Program(
    'run_approximation',
    [
        'run_approximation.cpp',
        'config.o',
        'io.o',
    ],
)

env.Program(
    'run_compute_iact',
    [
        'run_compute_iact.cpp',
        'acf.o',
        'config.o',
        'io.o',
    ],
)

env.Program(
    'run_generate_data',
    [
        'run_generate_data.cpp',
        'config.o',
        'io.o',
    ],
)

env.Program(
    'run_lr_zigzag',
    [
        'run_lr_zigzag.cpp',
        'acf.o',
        'config.o',
        'io.o',
        'lr_zigzag.o',
        'util.o',
    ],
)

env.Program(
    'run_sampler',
    [
        'run_sampler.cpp',
        'config.o',
        'flymc.o',
        'fmh.o',        
        'io.o',
        'mh.o',
    ],
)

env.Object(
    'target_test.o',
    'target_test.cpp',
)

env.Object(
    'util.o',
    'util.cpp',
)
