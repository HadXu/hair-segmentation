# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     __init__.py
   Description :
   Author :       haxu
   date：          2019-05-29
-------------------------------------------------
   Change Activity:
                   2019-05-29:
-------------------------------------------------
"""
__author__ = 'haxu'

from .pspnet import PSPNet


def get_network(name):
    name = name.lower()
    if name == 'pspnet_squeezenet':
        return PSPNet(num_class=1, base_network='squeezenet')
    elif name == 'pspnet_resnet101':
        return PSPNet(num_class=1, base_network='resnet101')
    raise ValueError
