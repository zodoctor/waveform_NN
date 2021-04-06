import pycbc
import numpy as np
import pycbc.filter

def calc_mismatch(y1_amp,y1_phase,y2_amp,y2_phase,delta_f,f_final,interp_grid):
    lin_interp_grid = np.linspace(0,f_final,int(f_final/delta_f))
    # interpolate onto linear grid
    y1_amp_interp = np.interp(lin_interp_grid,interp_grid,y1_amp)
    y2_amp_interp = np.interp(lin_interp_grid,interp_grid,y2_amp)
    y1_phase_interp = np.interp(lin_interp_grid,interp_grid,y1_phase)
    y2_phase_interp = np.interp(lin_interp_grid,interp_grid,y2_phase)
    
    # convert to complex frequency series
    y1_interp = y1_amp_interp*np.exp(1j*y1_phase_interp)
    y2_interp = y2_amp_interp*np.exp(1j*y2_phase_interp)
    y1_interp = pycbc.types.frequencyseries.FrequencySeries(y1_interp,delta_f)
    y2_interp = pycbc.types.frequencyseries.FrequencySeries(y2_interp,delta_f)
    # calc match and return
    return 1. - pycbc.filter.matchedfilter.match(y1_interp,y2_interp)[0]
