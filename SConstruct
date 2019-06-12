import hashlib
import json
import os

env = Environment(
    CPPFLAGS=[
	'-g',
        '-O2',
	'-std=c++14',
    ],
    CPPPATH=[
        Dir('.'),
        Dir('build'),
    ],
)

# Options for profiling.
if False:
    env.Append(CPPFLAGS='-pg')
    env.Append(LINKFLAGS='-pg')

test_alias = env.Alias('test')

def test_build_function(*args, **kwargs):
    # First positional arg is environment, second positional argument (if it
    # exists) is target.
    if kwargs and 'target' in kwargs:
        target = kwargs['target']
    else:
        target = args[1]
        
    executable_name = str(target) + '_exec'
    result_name = str(target) + '.result'

    # Place target as first positional argument, and passs all remaining args
    # to the Program builder.
    exec_args = len(args) > 2 and args[2:] or []
    exec_args = [executable_name] + list(exec_args)
    exec_kwargs = kwargs.copy()
    exec_kwargs.pop('target', None)
    
    test_executable = env.Program(*exec_args, **exec_kwargs)

    test_run = env.Command(
        target=result_name,
        source=executable_name,
        action=(
            os.path.join(Dir('.').path, str(test_executable[0]))
            + ' && touch ' + os.path.join(Dir('.').path, result_name)
        ),
    )

    Depends(test_run, test_executable)
    Depends(test_alias, test_run)    
    return test_run

env.AddMethod(test_build_function, 'Test')

env.ParseConfig(
'''if pkg-config --exists hdf5; then
  pkg-config --cflags --libs hdf5;
  else echo "-lhdf5";
fi'''
)
env.Append(LIBS='-lhdf5_cpp')

for pkg in ['eigen3',
            'gsl',
            'jsoncpp']:
    env.ParseConfig('pkg-config --cflags --libs %s' % pkg)

env.Append(LIBS='boost_program_options')
env.Append(LIBS='boost_timer')
env.Append(LIBS='boost_unit_test_framework')
    
SConscript(
    [
        'SConscript',
    ],
    exports='env',
    variant_dir='build',
)
