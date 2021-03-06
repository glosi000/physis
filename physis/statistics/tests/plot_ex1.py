#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#These are all the values of rho, energy, pressure and chemical pot. for the 
#systems with 216, 512, 1000 particles with the varying cut-off

T = np.array([0.9,1,1.15,1.25])

rho3g = np.array([0.013,0.029,0.073,0.139])
err3g = np.array([0.006,0.011,0.023,0.034])
rho3l = np.array([0.731,0.68,0.578,0.437])
err3l = np.array([0.016,0.027,0.032,0.04])

rho4g = np.array([0.017,0.029,0.092,0.134])
err4g = np.array([0.003,0.010,0.009,0.034])
rho4l = np.array([0.752,0.687,0.604,0.468])
err4l = np.array([0.009,0.011,0.013,0.033])

rho5g = np.array([0.023,0.029,0.082,0.14])
err5g = np.array([0.003,0.004,0.008,0.015])
rho5l = np.array([0.754,0.697,0.612,0.508])
err5l = np.array([0.007,0.011,0.010,0.017])

#errorbar(gas,T,xerr=errgas); errorbar(liq,T,xerr=errliq); savefig("phase_error.png")

E_3_g = np.array([-0.645,-2.966,-16.221,-38.258])
err_E_3_g = np.array([0.937,.562,21.219,18.747])
E_3_l = np.array([-1067.332,-954.993,-746.883,-541.703])
err_E_3_l = np.array([26.551,37.354,49.986,47.161])

E_4_g = np.array([-2.531,-7.167,-62.555,-102.06])
err_E_4_g = np.array([1.736,4.837,15.443,50.63])
E_4_l = np.array([-2617.431,-2312.666,-1783.812,-1346.913])
err_E_4_l = np.array([46.089,63.258,54.626,139.943])

E_5_g = np.array([-9.738,-13.964,-97.257,-244.096])
err_E_5_g = np.array([3.753,5.384,20.638,56.616])
E_5_l = np.array([-5071.84,-4608.068,-3609.889,-2728.973])
err_E_5_l = np.array([43.37,93.713,98.786,55.962])

chem_3_g = np.array([-4.14,-3.898,-3.71,-3.556])
err_chem_3_g = np.array([0.457,0.32,0.209,0.176])
chem_3_l = np.array([0.728,-2.073,-3.333,-3.481])
err_chem_3_l = np.array([6.522,3.362,0.948,0.453])

chem_4_g = np.array([-3.875,-3.895,-3.572,-3.614])
err_chem_4_g = np.array([0.15,0.274,0.079,0.115])
chem_4_l = np.array([-0.825,-3.281,-3.518,-3.593])
err_chem_4_l = np.array([3.974,1.361,0.6,0.297])

chem_5_g = np.array([-3.671,-3.853,-3.622,-3.584])
err_chem_5_g = np.array([0.099,0.109,0.06,0.07])
chem_5_l = np.array([-2.172,-3.493,-3.55,-3.541])
err_chem_5_l = np.array([2.56,0.953,0.437,0.266])

pres_3_g = np.array([0.011,0.025,0.061,0.106])
err_pres_3_g = np.array([0.006,0.013,0.031,0.067])
pres_3_l = np.array([0.031,0.038,0.065,0.114])
err_pres_3_l = np.array([0.279,0.265,0.227,0.164])

pres_4_g = np.array([0.014,0.025,0.069,0.097])
err_pres_4_g = np.array([0.004,0.01,0.023,0.04])
pres_4_l = np.array([0.007,0.018,0.064,0.09])
err_pres_4_l = np.array([0.186,0.173,0.16,0.114])

pres_5_g = np.array([0.018,0.025,0.065,0.099])
err_pres_5_g = np.array([0.004,0.006,0.015,0.025])
pres_5_l = np.array([0.052,0.021,0.065,0.106])
err_pres_5_l = np.array([0.15,0.134,0.11,0.1])
