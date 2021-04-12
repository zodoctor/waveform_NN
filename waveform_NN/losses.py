import tensorflow as tf

class MismatchLoss:
    def __init__(self,normalizer,freqs):
        self.normalizer=normalizer
        self.freqs = freqs
        self.n_freqs = len(self.freqs)
        self.dfs = tf.experimental.numpy.diff(self.freqs)

    def __call__(self,y_true,y_pred):
        
        # recolor data
        y_true = normalizer.color(y_true)
        y_pred = normalizer.color(y_pred)
        
        # unpack data
        hp_amp_true,hx_amp_true,hp_phase_true,hx_phase_true = y_true[:,:n_freqs], y_true[:,n_freqs:2*n_freqs], y_true[:,2*n_freqs:3*n_freqs], y_true[:,3*n_freqs:4*n_freqs]
        hp_amp_pred,hx_amp_pred,hp_phase_pred,hx_phase_pred = y_pred[:,:n_freqs], y_pred[:,n_freqs:2*n_freqs], y_pred[:,2*n_freqs:3*n_freqs], y_pred[:,3*n_freqs:4*n_freqs]

        # norms
        hp_norm_true = tf.math.sqrt(4.*np.sum((hp_amp_true[:,:-1]**2 + hp_amp_true[:,1:]**2)/2. * self.dfs))
        hx_norm_true = tf.math.sqrt(4.*np.sum((hx_amp_true[:,:-1]**2 + hx_amp_true[:,1:]**2)/2. * self.dfs))
        hp_norm_pred = tf.math.sqrt(4.*np.sum((hp_amp_pred[:,:-1]**2 + hp_amp_pred[:,1:]**2)/2. * self.dfs))
        hx_norm_pred = tf.math.sqrt(4.*np.sum((hx_amp_pred[:,:-1]**2 + hx_amp_pred[:,1:]**2)/2. * self.dfs))
        
        # phase differences
        cos_phase_diff_p = tf.math.cos(hp_phase_pred - hp_phase_true)
        cos_phase_diff_x = tf.math.cos(hx_phase_pred - hx_phase_true)
        
        # collect integrands
        integrand_p = hp_amp_true*hp_amp_pred*cos_phase_diff_p
        integrand_x = hx_amp_true*hx_amp_pred*cos_phase_diff_x 
        
        # riemann sum
        mismatch_p = 1. - 4.*np.sum(self.dfs*(integrand_p[:,-1] + integrand_p[:,1:])/2.)/(hp_norm_true*hp_norm_pred)
        mismatch_x = 1. - 4.*np.sum(self.dfs*(integrand_x[:,-1] + integrand_x[:,1:])/2.)/(hx_norm_true*hx_norm_pred)
        return mismatch_p + mismatch_x
        
