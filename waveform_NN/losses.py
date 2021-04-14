import tensorflow as tf

class MismatchLoss:
    def __init__(self,normalizer,freqs,approx=False):
        self.normalizer=normalizer
        self.freqs = freqs
        self.n_freqs = len(self.freqs)
        self.dfs = tf.experimental.numpy.diff(self.freqs)
        self.__name__ = 'mismatch'
        self.approx = approx

    def __call__(self,y_true,y_pred):
        print('yshapes:',y_true.shape,y_pred.shape) 
        # recolor data
        y_true = self.normalizer.color(y_true)
        y_pred = self.normalizer.color(y_pred)
        
        # unpack data
        hp_amp_true,hx_amp_true,hp_phase_true,hx_phase_true = y_true[:,:self.n_freqs], y_true[:,self.n_freqs:2*self.n_freqs], y_true[:,2*self.n_freqs:3*self.n_freqs], y_true[:,3*self.n_freqs:4*self.n_freqs]
        hp_amp_pred,hx_amp_pred,hp_phase_pred,hx_phase_pred = y_pred[:,:self.n_freqs], y_pred[:,self.n_freqs:2*self.n_freqs], y_pred[:,2*self.n_freqs:3*self.n_freqs], y_pred[:,3*self.n_freqs:4*self.n_freqs]

        # norms
        hp_norm_true = tf.math.sqrt(4.*tf.reduce_sum((hp_amp_true[:,:-1]**2 + hp_amp_true[:,1:]**2)/2. * self.dfs, 1))
        hx_norm_true = tf.math.sqrt(4.*tf.reduce_sum((hx_amp_true[:,:-1]**2 + hx_amp_true[:,1:]**2)/2. * self.dfs, 1))
        hp_norm_pred = tf.math.sqrt(4.*tf.reduce_sum((hp_amp_pred[:,:-1]**2 + hp_amp_pred[:,1:]**2)/2. * self.dfs, 1))
        hx_norm_pred = tf.math.sqrt(4.*tf.reduce_sum((hx_amp_pred[:,:-1]**2 + hx_amp_pred[:,1:]**2)/2. * self.dfs, 1))
        
        # phase differences
        if self.approx:
            
            cos_phase_diff_p = 1. - ((hp_phase_pred - hp_phase_true)**2)/2.
            cos_phase_diff_x = 1. - ((hx_phase_pred - hx_phase_true)**2)/2.
            integrand_p = tf.where(cos_phase_diff_p > 0, 
                tf.math.abs(hp_amp_true)*hp_amp_pred*cos_phase_diff_p,
                tf.math.abs(hp_amp_true)*tf.math.abs(hp_amp_pred)*cos_phase_diff_p
            )
            integrand_x = tf.where(cos_phase_diff_x > 0, 
                tf.math.abs(hx_amp_true)*hx_amp_pred*cos_phase_diff_x,
                tf.math.abs(hx_amp_true)*tf.math.abs(hx_amp_pred)*cos_phase_diff_x
            )
        else:
            cos_phase_diff_p = tf.math.cos(hp_phase_pred - hp_phase_true)
            cos_phase_diff_x = tf.math.cos(hx_phase_pred - hx_phase_true)
            # collect integrands
            integrand_p = tf.math.abs(hp_amp_true)*tf.math.abs(hp_amp_pred)*cos_phase_diff_p
            integrand_x = tf.math.abs(hx_amp_true)*tf.math.abs(hx_amp_pred)*cos_phase_diff_x 
        
        # riemann sum
        mismatch_p = 1. - 4.*tf.reduce_sum(self.dfs*(integrand_p[:,:-1] + integrand_p[:,1:])/2., 1)/(hp_norm_true*hp_norm_pred)
        mismatch_x = 1. - 4.*tf.reduce_sum(self.dfs*(integrand_x[:,:-1] + integrand_x[:,1:])/2., 1)/(hx_norm_true*hx_norm_pred)
        return mismatch_p + mismatch_x

class ZosmatchLoss:
    def __init__(self,normalizer,freqs,approx=False):
        self.normalizer=normalizer
        self.freqs = freqs
        self.n_freqs = len(self.freqs)
        self.dfs = tf.experimental.numpy.diff(self.freqs)
        self.__name__ = 'mismatch'
        self.approx = approx

    def __call__(self,y_true,y_pred):
        print('yshapes:',y_true.shape,y_pred.shape) 
        # recolor data
        y_true = self.normalizer.color(y_true)
        y_pred = self.normalizer.color(y_pred)
        
        # unpack data
        hp_amp_true,hx_amp_true,hp_phase_true,hx_phase_true = y_true[:,:self.n_freqs], y_true[:,self.n_freqs:2*self.n_freqs], y_true[:,2*self.n_freqs:3*self.n_freqs], y_true[:,3*self.n_freqs:4*self.n_freqs]
        hp_amp_pred,hx_amp_pred,hp_phase_pred,hx_phase_pred = y_pred[:,:self.n_freqs], y_pred[:,self.n_freqs:2*self.n_freqs], y_pred[:,2*self.n_freqs:3*self.n_freqs], y_pred[:,3*self.n_freqs:4*self.n_freqs]

        phase_diff_sqrd_p = tf.math.square(hp_phase_pred - hp_phase_true)/2.
        phase_diff_sqrd_x = tf.math.square(hx_phase_pred - hx_phase_true)/2.

        return tf.reduce_sum(tf.math.abs(hp_amp_true-hp_amp_pred) + 1000.*phase_diff_sqrd_p + tf.math.abs(hx_amp_true-hx_amp_pred) + 1000.*phase_diff_sqrd_x, 1)
        
