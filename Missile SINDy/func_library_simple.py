import numpy as np
import pysindy as ps

library_functions = [
    # lambda x: np.exp(x),
    lambda x: x,
    lambda x: 1. / x,
    lambda x: np.sin(x),
    lambda x: np.cos(x),

    lambda x, y: x * y,
    lambda x, y: x / y,
    lambda x, y: y / x,

    lambda x, y: x * np.cos(y),
    lambda x, y: y * np.cos(x),
    # lambda x, y: np.cos(y) / x,
    # lambda x, y: np.cos(x) / y,

    lambda x, y: x * np.sin(y),
    lambda x, y: y * np.sin(x),
    # lambda x, y: np.sin(y) / x,
    # lambda x, y: np.sin(x) / y,

    lambda x, y, z: x * y / z,
    lambda x, y, z: x * z / y,
    lambda x, y, z: z * y / x,

    lambda x, y: np.sqrt(x ** 2 + y ** 2),
    lambda x, y: np.sqrt(x ** 2 + y ** 2) * np.arctan(y / x),
    lambda x, y: x ** 2 + y ** 2

    # lambda x, y, z: x * np.cos(y) / z,
    # lambda x, y, z: x * np.cos(z) / y,
    # lambda x, y, z: y * np.cos(x) / z,
    # lambda x, y, z: y * np.cos(z) / x,
    # lambda x, y, z: z * np.cos(y) / x,
    # lambda x, y, z: z * np.cos(x) / y,
    #
    # lambda x, y, z: x * np.sin(y) / z,
    # lambda x, y, z: x * np.sin(z) / y,
    # lambda x, y, z: y * np.sin(x) / z,
    # lambda x, y, z: y * np.sin(z) / x,
    # lambda x, y, z: z * np.sin(y) / x,
    # lambda x, y, z: z * np.sin(x) / y,

]
library_function_names = [
    # lambda x: 'exp(' + x + ')',
    lambda x: x,
    lambda x: '1 / ' + x,
    lambda x: 'sin( ' + x + ' ) ',
    lambda x: 'cos( ' + x + ' )',

    lambda x, y: x + ' * ' + y,
    lambda x, y: x + ' / ' + y,
    lambda x, y: y + ' / ' + x,

    lambda x, y: x + ' * cos( ' + y + ' )',
    lambda x, y: y + ' * cos( ' + x + ' )',
    # lambda x, y: 'cos( ' + y + ' ) / ' + x,
    # lambda x, y: 'cos( ' + x + ' ) / ' + y,

    lambda x, y: x + ' * sin( ' + y + ' )',
    lambda x, y: y + ' * sin( ' + x + ' )',
    # lambda x, y: 'sin( ' + y + ' ) / ' + x,
    # lambda x, y: 'sin( ' + x + ' ) / ' + y,

    lambda x, y, z: x + ' * ' + y + ' / ' + z,
    lambda x, y, z: x + ' * ' + z + ' / ' + y,
    lambda x, y, z: z + ' * ' + y + ' / ' + x,

    lambda x, y: 'sqrt(' + x + '**2 + ' + y + '**2)',
    lambda x, y: 'sqrt(' + x + '**2 + ' + y + '**2)*arctan(' + y + '/' + x + ')',
    lambda x, y: x + '**2 + ' + y + '**2'

    # lambda x, y, z: x + ' * cos( ' + y + ' ) / ' + z,
    # lambda x, y, z: x + ' * cos( ' + z + ' ) / ' + y,
    # lambda x, y, z: y + ' * cos( ' + x + ' ) / ' + z,
    # lambda x, y, z: y + ' * cos( ' + z + ' ) / ' + x,
    # lambda x, y, z: z + ' * cos( ' + y + ' ) / ' + x,
    # lambda x, y, z: z + ' * cos( ' + x + ' ) / ' + y,
    #
    # lambda x, y, z: x + ' * sin( ' + y + ' ) / ' + z,
    # lambda x, y, z: x + ' * sin( ' + z + ' ) / ' + y,
    # lambda x, y, z: y + ' * sin( ' + x + ' ) / ' + z,
    # lambda x, y, z: y + ' * sin( ' + z + ' ) / ' + x,
    # lambda x, y, z: z + ' * sin( ' + y + ' ) / ' + x,
    # lambda x, y, z: z + ' * sin( ' + x + ' ) / ' + y
]
custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)