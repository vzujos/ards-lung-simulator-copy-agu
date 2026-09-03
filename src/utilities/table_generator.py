# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:32:58 2026

@author: angus
"""

import numpy as np

calibrated_params = {"PIG2":{"c_tissue": 1.2177, # 0.013 medium-fine
                             "K_cw": 0.1037,
                             "K_d": 0.0240,
                             "alpha": 10.9497,
                             "Gamma Insp": 0.2369,
                             "Gamma Exp": 0.4452,},
                     "PIG3":{ "c_tissue": 1.6014, # 0.028 medium-fine
                              "K_cw": 0.1000,
                              "K_d": 0.1000,
                              "alpha": 11.0346,
                              "Gamma Insp": 0.1872,
                              "Gamma Exp": 0.3126,},
                     "PIG4":{"c_tissue": 1.2266, # 0.009 medium
                             "K_cw": 0.0509, 
                             "K_d": 0.0103,
                             "alpha": 10.4279,
                             "Gamma Insp": 0.4716,
                             "Gamma Exp": 0.6196,},
                     "PIG5":{"c_tissue": 1.5047, # 0.015 medium
                             "K_cw": 0.0204,
                             "K_d": 0.0767,
                             "alpha": 10.5071,
                             "Gamma Insp": 0.2945,
                             "Gamma Exp": 0.4875,},
                     "PIG6":{"c_tissue": 1.6788, # 0.025 medium-fine
                             "K_cw": 0.1086,
                             "K_d": 0.1037,
                             "alpha": 10.3918,
                             "Gamma Insp": 0.1536,
                             "Gamma Exp": 0.3665,},
                       }

for param in ["c_tissue", "K_cw","K_d","alpha","Gamma Insp","Gamma Exp"]:
    data = []
    for pig in [2,3,4,5,6]:
        data += [calibrated_params["PIG%i"%pig][param]]
    
    if param in ["c_tissue", "Gamma Insp", "Gamma Exp", "alpha"]:
        fmt = "%.2f"
    else:
        fmt = "%.3f"
        
    print(param+": "+fmt%np.mean(data)+" ("+fmt%np.std(data)+")")