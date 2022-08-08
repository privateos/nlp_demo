from importlib import import_module
v = input()
# x = import_module('utils')
x = import_module(v)
x.my_print()