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

_key_pattern = re.compile(r'^[A-Za-z0-9\.\-_]+\=')

def _parse_key_value(key, next_value, args):
    if _key_pattern.match(key):
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

def load_yaml(config_file):
    import yaml
    loaded_data = {}
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
        for section, settings in config.items():
            if settings is not None:
                for key, value in settings.items():
                    loaded_data[key] = value
        return loaded_data

def load_config(config_file):
    if config_file.endswith('.yaml'):
        return load_yaml(config_file)
    return {}

class AdhocArguments(object):
    """
    アドホックな引数パラメータ
    """
    def __init__(self, args:dict, expand_config=None, default_args:dict=None, use_environ=True):
        self._args = {}
        self._used_keys = set()
        self._use_environ = use_environ
        for key, value in args.items():
            if key == expand_config:
                self.load_config(value)
            else:
                self.args[key] = value
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
                    self._used_keys.add(key)
                    args[key] = value
            if lost_found:
                sys.exit(1)

    def __repr__(self):
        return repr(self._args)

    def __getitem__(self, key):
        keys = key.split('|')
        for key in keys:
            if key in self._args:
                self._used_keys.add(key)
                return self._args[key]
            if key.startswith('='):
                return parse_argument_value(key[1:])
            if key.startswith('!'):
                raise ValueError(f'{key[0]} is unset.')
            if self._use_environ:
                environ_key = key.upper()
                if environ_key in os.environ:
                    value = parse_argument_value(os.environ[environ_key])
                    self._used_keys.add(key)
                    self._args[key] = value
                    return value
        return None

    def __setitem__(self, key, value):
        self._args[key] = value
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self._args

    def load_config(self, config_file, merge_data=True):
        loaded_data = load_config(config_file)
        if merge_data:
            self._args.update(loaded_data)
        return loaded_data

    def utils_check(self):
        show_notion = True
        for key, value in self._args.items():
            if key not in self._used_keys:
                if show_notion:
                    self.utils_print(f'未使用のパラメータ一覧//List of unused parameters')
                    show_notion = False
                print(f'{key}: {repr(value)}')
        if not show_notion:
            self.utils_print(f'スペルミスがないか確認してください//Check if typos exist.')

    def raise_uninstalled_module(self, module_name):
        self.utils_print(f'{module_name}がインストールされていません//Uninstalled {module_name}')
        print(f'pip3 install -U {module_name}')
        sys.exit(1)

    def raise_unset_key(self, key, desc_ja=None, desc_en=None):
        desc_ja = f' ({desc_ja})' if desc_ja else ''
        desc_en = f' ({desc_en})' if desc_en else ''
        self.utils_print(f'{key}{desc_ja}を設定してください//Please set {key}{desc_en}')
        sys.exit(1)

    def utils_print(self, *args, **kwargs):
        print("🐥", *args, **kwargs)

    def verbose_print(self, *args, **kwargs):
        print("🐓", *args, **kwargs)

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