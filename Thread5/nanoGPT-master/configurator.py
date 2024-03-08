"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

def safe_eval(val):
    try:
        return literal_eval(val)
    except (SyntaxError, ValueError):
        return val

for arg in sys.argv[1:]:
    if '=' not in arg and not arg.startswith('--'):
        # This is assumed to be a config file name
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            exec(f.read(), globals())
    elif '=' in arg:
        # This is assumed to be a --key=value argument
        key, val = arg.split('=', 1)
        key = key[2:]  # Remove leading '--'
        val = safe_eval(val)  # Try to evaluate the value
        if key in globals():
            if isinstance(val, type(globals()[key])):
                print(f"Overriding: {key} = {val}")
                globals()[key] = val
            else:
                print(f"Warning: Type mismatch for '{key}'. Expected {type(globals()[key])}, got {type(val)}. Using the original value.")
        else:
            print(f"Warning: Unknown config key: {key}. Ignoring.")
    else:
        # If the argument starts with '--' but doesn't contain '=', it's an invalid format
        if arg.startswith('--'):
            raise ValueError(f"Invalid argument format. Expected '--key=value'. Given: {arg}")
        else:
            raise ValueError(f"Config file name should not be prefixed with '--'. Given: {arg}")


