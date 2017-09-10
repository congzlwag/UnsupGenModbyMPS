# -*- coding: utf-8 -*-
"""Simply gauge transform a trained MPS"""
import sys
sys.path.append('/home/qi/paperAllNew/')
from MPScumulant import MPS_c
m = MPS_c(28**2)
m.loadMPS('Loop150MPS/')
m.verbose=0
while m.current_bond > 28*14:
    m.merge_bond()
    m.rebuild_bond(False,kepbdm=True)
m.saveMPS('Loop150MPS_middle')