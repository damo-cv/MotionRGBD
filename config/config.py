'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import yaml
# from easydict import EasyDict as edict
def Config(args):
    print()
    print('='*80)
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for dic in config:
        for k, v in config[dic].items():
            setattr(args, k, v)
            print(k, ':\t', v)
    print('='*80)
    print()
    return args