from __future__ import absolute_import

from tframe import console

try:
  from Tkinter import *
  console.show_status('Tkinter imported')
except:
  from tkinter import *
  console.show_status('tkinter imported')


if __name__ == 'main':
  pass