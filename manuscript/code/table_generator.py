# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:32:58 2026

@author: angus
"""

import numpy as np


calibrated_params = {"PIG2":{"c_tissue": 1.5119, # 0.013 medium-fine
                             "K_cw": 0.0914,
                             "K_d": 0.0100,
                             "alpha": 11.200,
                             "Gamma Insp": 0.2082,
                             "Gamma Exp": 0.3731,},
                     "PIG3":{ "c_tissue": 1.9496, # 0.028 medium-fine
                              "K_cw": 0.0487,
                              "K_d": 0.0962,
                              "alpha": 11.9908,
                              "Gamma Insp": 0.1888,
                              "Gamma Exp": 0.3122,},
                     "PIG4":{"c_tissue": 1.2817, # 0.009 medium
                             "K_cw": 0.0540, 
                             "K_d": 0.0099,
                             "alpha": 10.8658,
                             "Gamma Insp": 0.4797,
                             "Gamma Exp": 0.6484,},
                     "PIG5":{"c_tissue": 1.5975, # 0.015 medium
                             "K_cw": 0.0166,
                             "K_d": 0.0675,
                             "alpha": 11.000,
                             "Gamma Insp": 0.3183,
                             "Gamma Exp": 0.4433,},
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
        
    std = np.std(data)
    mean=  np.mean(data)
    cv =  std/mean
    print(param+": "+fmt%mean+" ("+fmt%std+")" + "[%.2f]"%cv)