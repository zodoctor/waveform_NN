import pycbc
import multiprocessing
from pycbc.waveform import td_approximants, fd_approximants
import numpy as np
import h5py

def get_single_waveform(
    sample,
    distance=1e-20,
    f_lower=10.,
    delta_f=1./32,
    f_final=1000.,
    approximant='IMRPhenomXPHM',
    n_freq_interp = 500,
    freq_interp_type = 'log'
    ):
    """
    get an interpolated single waveform.

    Parameters
    ----------
    sample: dict or line of np structured array
        must have names for parameters as keys like mass1, mass,2, ...
    distance: float
        distance in Mpc
    f_lower: float
        low frequency in Hz
    delta_f: float
        frequency spacing in waveform generation
    f_final: float
        final frequency in Hz
    approximant: str
        approximant to use in freq generatoin
    """
    if isinstance(sample,dict):
        samp_dict = sample
    else:
        samp_dict = {name:sample[name] for name in sample.dtype.names}
    hp, hc = pycbc.waveform.get_fd_waveform(approximant=approximant,distance=distance,
                             f_lower=f_lower, delta_f=delta_f,f_final=f_final,**samp_dict)

    x = [samp_dict['mass2']/samp_dict['mass1'],
              samp_dict['spin_mag_1'],samp_dict['spin_mag_2'],
              samp_dict['cos_az_ang_1'],samp_dict['cos_az_ang_2'],
              samp_dict['pol_ang_1'],samp_dict['pol_ang_2'],
              samp_dict['inclination']]
    if freq_interp_type == 'log':
        interp_grid = np.logspace(np.log10(f_lower),np.log10(f_final),n_freq_interp)
    else:
        raise NotImplementedError

    hp_amp_interp = np.interp(interp_grid,hp.sample_frequencies,np.sqrt(hp.squared_norm()))
    hc_amp_interp = np.interp(interp_grid,hc.sample_frequencies,np.sqrt(hc.squared_norm()))
    hp_phase_interp = np.interp(interp_grid,hp.sample_frequencies,np.unwrap(np.angle(hp)))
    hc_phase_interp = np.interp(interp_grid,hc.sample_frequencies,np.unwrap(np.angle(hc)))
    y = np.concatenate((hp_amp_interp,hc_amp_interp,hp_phase_interp,hc_phase_interp))
    return x,y

def generate_waveforms(outstr='wfsamples',nsamples=2**11,seed=1234,ncores=1,sample_type='random',return_XY=False):
    np.random.seed(seed)

    samples_dict = {}
    total_mass = 60
    
    if sample_type == 'random':
        mass_ratios = np.random.uniform(0.1,1,size=nsamples)

        #total_masses = np.random.uniform(5,100,size=nsamples)
        
        samples_dict['mass1'] = total_mass/(1.+mass_ratios)
        samples_dict['mass2'] = total_mass-samples_dict['mass1']

        spin_mags = np.random.uniform(0,1,size=nsamples*2)
        cos_az_ang = np.random.uniform(-1,1,size=nsamples*2)
        pol_ang = np.random.uniform(0,2.*np.pi,size=nsamples*2)
        in_plane_comp = spin_mags*np.sqrt(1.-cos_az_ang**2)
        sx = in_plane_comp*np.sin(pol_ang)
        sy = in_plane_comp*np.cos(pol_ang)
        sz = spin_mags*cos_az_ang
        samples_dict['spin_mag_1'],samples_dict['spin_mag_2'] = spin_mags[:nsamples],spin_mags[nsamples:]
        samples_dict['cos_az_ang_1'],samples_dict['cos_az_ang_2'] = cos_az_ang[:nsamples],cos_az_ang[nsamples:]
        samples_dict['pol_ang_1'],samples_dict['pol_ang_2'] = pol_ang[:nsamples],pol_ang[nsamples:]
        samples_dict['s1x'],samples_dict['s2x'] = sx[:nsamples],sx[nsamples:]
        samples_dict['s1y'],samples_dict['s2y'] = sy[:nsamples],sy[nsamples:]
        samples_dict['s1z'],samples_dict['s2z'] = sz[:nsamples],sz[nsamples:]
        samples_dict['inclination'] = np.arccos(np.random.uniform(-1,1,size=nsamples))
        
    elif sample_type=='lhs':
        
        # latin hypercube sampling
        samples = np.array([np.random.permutation(np.linspace(0,1,nsamples)) for i in range(8)])
        
        # masses / mass ratios
        mass_ratios = samples[0,:] - 1. # log mass ratios
        mass_ratios = 10**mass_ratios # unlog it
        #mass_ratios = samples[0,:]*0.9 + 0.1 # from 0.1 to 1
        samples_dict['mass1'] = total_mass/(1.+mass_ratios)
        samples_dict['mass2'] = total_mass-samples_dict['mass1']
        
        # spins
        samples_dict['spin_mag_1'],samples_dict['spin_mag_2'] = samples[1,:],samples[2,:]
        samples_dict['cos_az_ang_1'],samples_dict['cos_az_ang_2'] = 2.*samples[3,:] - 1 , 2.*samples[4,:] - 1
        samples_dict['pol_ang_1'],samples_dict['pol_ang_2'] = 2.*np.pi*samples[5,:], 2.*np.pi*samples[6,:]
        
        # inclination
        samples_dict['inclination'] = np.pi*samples[7,:]
        
    samples = np.array(list(zip(*[samples for _,samples in samples_dict.items()])),dtype=[(name,float) for name,_ in samples_dict.items()])


    if ncores==1:
        XYout = list(map(get_single_waveform,samples))
    else:
        print(f'multiprocessing with {ncores} cores')
        with multiprocessing.Pool(ncores) as pool:
            XYout = list(pool.map(get_single_waveform,samples,chunksize=len(samples)//ncores))


    X,Y = [XYout_line[0] for XYout_line in XYout], [XYout_line[1] for XYout_line in XYout]
    
    X = np.array(X)
    Y = np.array(Y)
    
    with h5py.File(f'{outstr}.h5','w') as f:
        xdset = f.create_dataset("X",data=X)
        ydset = f.create_dataset("Y",data=Y)

    #np.save(f'{outstr}_X.npy',X)
    #np.save(f'{outstr}_Y.npy',Y)
    if return_XY:
        return X,Y
