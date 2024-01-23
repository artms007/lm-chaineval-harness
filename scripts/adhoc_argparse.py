import os
import sys
import re

def parse_argument_value(value):
    try:
        return int(value)
    except:
        pass
    try:
        return float(value)
    except:
        pass
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    return value

key_pattern = re.compile(r'^[A-Za-z0-9\.\-_]+\=')

def _parse_key_value(key, next_value, args):
    if key_pattern.match(key):
        if key.startswith('--'):
            key = key[2:]
        key, _, value = key.partition('=')
        return key, parse_argument_value(value)     
    elif key.startswith('--'):
        key = key[2:]
        if next_value.startswith('--'):
            if key.startswith('enable_') or key.startswith('enable-'):
                return key[7:], True
            elif key.startswith('disable_') or key.startswith('disable-'):
                return key[8:], False
            return key, True
        else:
            args['_'] = next_value
            return key, parse_argument_value(next_value)
    else:
        if args.get('_') != key:
            files = args.get('files', [])
            files.append(key)
            args['files'] = files
    return key, None

class AdhocArguments(object):
    def __init__(self, args:dict, default_args:dict=None, use_environ=True):
        self._args = args
        self._use_environ = use_environ
        if default_args:
            lost_found = False
            for key, value in default_args.items():
                if key in args:
                    continue
                if use_environ:
                    environ_key = key.upper()
                    if environ_key in os.environ:
                        value = parse_argument_value(os.environ[environ_key])
                if isinstance(value, tuple) and len(value)==1:
                    print(f'Option {key} is required. {value[0]}')
                    lost_found = True
                else:
                    args[key] = value
            if lost_found:
                sys.exit(1)
        for key, value in args.items():
            setattr(self, key, value)

    def __repr__(self):
        return repr(self._args)

    def __getitem__(self, key):
        keys = key.split('|')
        for key in keys:
            if key in self._args:
                return self._args[key]
        if self._use_environ:
            for key in keys:
                environ_key = key.upper()
                if environ_key in os.environ:
                    value = parse_argument_value(os.environ[environ_key])
                    self._args[key] = value
                    return value
        return None

    def __setitem__(self, key, value):
        self._args[key] = value
        setattr(self, key, value)

def adhoc_argument_parser(default_args:dict=None, use_environ=True, load_config=None):
    argv = sys.argv[1:]
    args={'_': ''}
    for arg, next_value in zip(argv, argv[1:] + ['--']):
        key, value = _parse_key_value(arg, next_value, args)
        if value is not None:
            args[key.replace('-', '_')] = value
    del args['_']
    if load_config and load_config in args:
        ## TODO: Integrate YAML Config
        ...
    return AdhocArguments(args, default_args=default_args, use_environ=use_environ)

if __name__ == '__main__':
    args = adhoc_argument_parser()
    print(args)