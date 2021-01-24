#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#These are all the values of rho, energy, pressure and chemical pot. for the 
#systems with 512, 1000 particles with the constant cut-off simulations

T = np.array([0.9,1,1.15,1.25])

rho4g = np.array([0.012,0.023,0.066,0.168])
err4g = np.array([0.004,0.006,0.010,0.023])
rho4l = np.array([0.739,0.684,0.571,0.505])
err4l = np.array([0.009,0.015,0.016,0.026])

rho5g = np.array([0.013,0.031,0.076,0.142])
err5g = np.array([0.002,0.004,0.008,0.013])
rho5l = np.array([0.743,0.703,0.599,0.512])
err5l = np.array([0.005,0.008,0.011,0.014])

#errorbar(gas,T,xerr=errgas); errorbar(liq,T,xerr=errliq); savefig("phase_error.png")

E_4_g = np.array([-1.217,-4.14,-28.453,-191.437])
err_E_4_g = np.array([1.13,2.427,9.892,53.742])
E_4_l = np.array([-2594.598,-2335.438,-1803.23,-1272.945])
err_E_4_l = np.array([41.927,52.589,64.394,119.043])

E_5_g = np.array([-2.957,-15.765,-80.805,-261.856])
err_E_5_g = np.array([1.906,5.242,18.643,51.589])
E_5_l = np.array([-5013.878,-4637.204,-3587.524,-2719.258])
err_E_5_l = np.array([38.824,78.069,89.781,125.461])

chem_4_g = np.array([-4.157,-4.039,-3.736,-3.542])
err_chem_4_g = np.array([0.222,0.201,0.101,0.11])
chem_4_l = np.array([-2.438,-3.399,-3.618,-3.501])
err_chem_4_l = np.array([2.758,1.221,0.493,0.375])

chem_5_g = np.array([-4.071,-3.807,-3.663,-3.584])
err_chem_5_g = np.array([0.145,0.081,0.066,0.068])
chem_5_l = np.array([-3.115,-3.433,-3.624,-3.553])
err_chem_5_l = np.array([1.553,0.835,0.375,0.251])

pres_4_g = np.array([0.01,0.02,0.057,0.17])
err_pres_4_g = np.array([0.004,0.007,0.018,0.043])
pres_4_l = np.array([-0.03,-0.008,0.048,0.101])
err_pres_4_l = np.array([0.179,0.178,0.146,0.132])

pres_5_g = np.array([0.011,0.026,0.061,0.1])
err_pres_5_g = np.array([0.003,0.006,0.014,0.027])
pres_5_l = np.array([-0.017,0.033,0.078,0.093])
err_pres_5_l = np.array([0.128,0.120,0.109,0.093])
