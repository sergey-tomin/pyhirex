import numpy as np

q_e = 1.6021766208e-19       # C - Elementary charge
h_J_s = 6.626070040e-34      # Plancks constant [J*s]
h_eV_s = h_J_s / q_e                     # [eV*s]
hr_eV_s = h_eV_s/2./np.pi
speed_of_light = 299792458.0 # m/s

def find_nearest_idx(array, value):
    if value == -np.inf:
        value = np.amin(array)
    if value == np.inf:
        value = np.amax(array)
    return (np.abs(array-value)).argmin()

def calc_ph_sp_dens(spec, freq_ev, n_photons, spec_squared=1):
    """
    calculates number of photons per electronvolt
    """
    # _logger.debug('spec.shape = {}'.format(spec.shape))
    if spec.ndim == 1:
        axis = 0
    else:
        if spec.shape[0] == freq_ev.shape[0]:
            spec = spec.T
        axis = 1
        #     axis=0
        # elif spec.shape[1] == freq_ev.shape[0]:
        #     axis=1
        # else:
        #     raise ValueError('operands could not be broadcast together with shapes ', spec.shape, ' and ', freq_ev.shape)
    # _logger.debug('spec.shape = {}'.format(spec.shape))

    if spec_squared:
        spec_sum = np.trapz(spec, x=freq_ev, axis=axis)
    else:
        spec_sum = np.trapz(abs(spec) ** 2, x=freq_ev, axis=axis)

    if np.size(spec_sum) == 1:
        if spec_sum == 0:  # avoid division by zero
            spec_sum = np.inf
    else:
        spec_sum[spec_sum == 0] = np.inf  # avoid division by zero

    if spec_squared:
        norm_factor = n_photons / spec_sum
    else:
        norm_factor = np.sqrt(n_photons / spec_sum)

    if spec.ndim == 2:
        norm_factor = norm_factor[:, np.newaxis]
    # _logger.debug('spec.shape = {}'.format(spec.shape))
    # _logger.debug('norm_factor.shape = {}'.format(norm_factor.shape))
    spec = spec * norm_factor
    if axis == 1:
        spec = spec.T
    # _logger.debug('spec.shape = {}'.format(spec.shape))
    return spec
    

def imitate_1d_sase_like(td_scale, td_env, fd_scale, fd_env, td_phase=None, fd_phase=None, phen0=None, en_pulse=None,
                         fit_scale='td', n_events=1, **kwargs):
    """
    Models FEL pulse(s) based on Gaussian statistics
    td_scale - scale of the pulse on time domain [m]
    td_env - expected pulse envelope in time domain [W]
    fd_scale - scale of the pulse in frequency domain [eV]
    fd_env - expected pulse envelope in frequency domain [a.u.]
    td_phase - additional phase chirp to be added in time domain
    fd_phase - additional phase chirp to be added in frequency domain
    phen0 - sampling wavelength expressed in photon energy [eV]
    en_pulse - expected average energy of the pulses [J]
    fit_scale - defines the scale in which outputs should be returned:
        'td' - time domain scale td_scale is used for the outputs, frequency domain phase and envelope will be re-interpolated
        'fd' - frequency domain scale fd_scale is used for the outputs, time domain phase and envelope will be re-interpolated
    n_events - number of spectra to be generated

    returns tuple of 4 arguments: (ph_en, fd, s, td)
    fd_scale - colunm of photon energies in eV
    fd - matrix of radiation in frequency domain with shape, normalized such that np.sum(abs(fd)**2) is photon spectral density, i.e: np.sum(abs(fd)**2)*fd_scale = N_photons
    td - matrix of radiation in time domain, normalized such that abs(td)**2 = radiation_power in [w]
    """

    # _logger.info('generating 1d radiation field imitating SASE')
    
    seed = kwargs.get('seed', None)
    if seed is not None:
        np.random.seed(seed)

    if fit_scale == 'td':

        n_points = len(td_scale)
        s = td_scale
        Ds = (td_scale[-1] - td_scale[0])
        ds = Ds / n_points

        td = np.random.randn(n_points, n_events) + 1j * np.random.randn(n_points, n_events)
        td *= np.sqrt(td_env[:, np.newaxis])
        fd = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(td, axes=0), axis=0), axes=0)
        # fd = np.fft.ifft(td, axis=0)
        # fd = np.fft.fftshift(fd, axes=0)

        if phen0 is not None:
            e_0 = phen0
        else:
            e_0 = np.mean(fd_scale)

        # internal interpolated values
        fd_scale_i = h_eV_s * np.fft.fftfreq(n_points, d=(
                ds / speed_of_light)) + e_0  # internal freq.domain scale based on td_scale
        fd_scale_i = np.fft.fftshift(fd_scale_i, axes=0)
        fd_env_i = np.interp(fd_scale_i, fd_scale, fd_env, right=0, left=0)

        if fd_phase is None:
            fd_phase_i = np.zeros_like(fd_env_i)
        else:
            fd_phase_i = np.interp(fd_scale_i, fd_scale, fd_phase, right=0, left=0)

        fd *= np.sqrt(fd_env_i[:, np.newaxis]) * np.exp(1j * fd_phase_i[:, np.newaxis])

        # td = np.fft.ifftshift(fd, axes=0)
        # td = np.fft.fft(td, axis=0)
        td = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(fd, axes=0), axis=0), axes=0)

        td_scale_i = td_scale

    elif fit_scale == 'fd':

        n_points = len(fd_scale)
        Df = abs(fd_scale[-1] - fd_scale[0]) / h_eV_s
        df = Df / n_points

        fd = np.random.randn(n_points, n_events) + 1j * np.random.randn(n_points, n_events)
        fd *= np.sqrt(fd_env[:, np.newaxis])
        td = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(fd, axes=0), axis=0), axes=0)

        td_scale_i = np.fft.fftfreq(n_points, d=df) * speed_of_light
        td_scale_i = np.fft.fftshift(td_scale_i, axes=0)
        td_scale_i -= np.amin(td_scale_i)
        td_env_i = np.interp(td_scale_i, td_scale, td_env, right=0, left=0)

        if td_phase is None:
            td_phase_i = np.zeros_like(td_env_i)
        else:
            td_phase_i = np.interp(td_scale_i, td_scale, td_phase, right=0, left=0)

        td *= np.sqrt(td_env_i[:, np.newaxis]) * np.exp(1j * td_phase_i[:, np.newaxis])

        fd = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(td, axes=0), axis=0), axes=0)

        fd_scale_i = fd_scale

    else:
        raise ValueError('fit_scale should be either "td" of "fd"')

    # # normalization for pulse energy
    # if en_pulse == None:
        # # _logger.debug(ind_str + 'no en_pulse provided, calculating from integral of td_env')
        # en_pulse = np.trapz(td_env, td_scale / speed_of_light)

    pulse_energies = np.trapz(abs(td) ** 2, td_scale_i / speed_of_light, axis=0)
    # scale_coeff = en_pulse / np.mean(pulse_energies)
    # td *= np.sqrt(scale_coeff)

    # normalization for photon spectral density
    spec = np.mean(np.abs(fd) ** 2, axis=1)
    spec_center = np.sum(spec * fd_scale_i) / np.sum(spec)

    n_photons = pulse_energies / q_e / spec_center
    fd = calc_ph_sp_dens(fd, fd_scale_i, n_photons, spec_squared=0)
    td_scale, fd_scale = td_scale_i, fd_scale_i

    np.random.seed()

    return (td_scale, td, fd_scale, fd)


def imitate_1d_sase(spec_center=500, spec_res=0.01, spec_width=2.5, spec_range=(None, None), pulse_length=6,
                    en_pulse=1e-3, flattop=0, n_events=1, spec_extend=5, **kwargs):
    """
    Models FEL pulse(s) based on Gaussian statistics
    spec_center - central photon energy in eV
    spec_res - spectral resolution in eV
    spec_width - width of spectrum in eV (fwhm of E**2)
    spec_range = (E1, E2) - energy range of the spectrum. If not defined, spec_range = (spec_center - spec_width * spec_extend, spec_center + spec_width * spec_extend)
    pulse_length - longitudinal size of the pulse in um (fwhm of E**2)
    en_pulse - expected average energy of the pulses in Joules
    flattop - if true, flat-top pulse in time domain is generated with length 'pulse_length' in um
    n_events - number of spectra to be generated

    return tuple of 4 arguments: (s, td, ph_en, fd)
    ph_en - colunm of photon energies in eV with size (spec_range[2]-spec_range[1])/spec_res
    fd - matrix of radiation in frequency domain with shape ((spec_range[2]-spec_range[1])/spec_res, n_events), normalized such that np.sum(abs(fd)**2) is photon spectral density, i.e: np.sum(abs(fd)**2)*spec_res = N_photons
    s - colunm of longitudinal positions along the pulse in yime domain in um
    td - matrix of radiation in time domain with shape ((spec_range[2]-spec_range[1])/spec_res, n_events), normalized such that abs(td)**2 = radiation_power
    """

    if spec_range == (None, None):
        spec_range = (spec_center - spec_width * spec_extend, spec_center + spec_width * spec_extend)
    elif spec_center == None:
        spec_center = (spec_range[1] + spec_range[0]) / 2

    pulse_length_sigm = pulse_length / (2 * np.sqrt(2 * np.log(2)))
    spec_width_sigm = spec_width / (2 * np.sqrt(2 * np.log(2)))

    fd_scale = np.arange(spec_range[0], spec_range[1], spec_res)
    n_points = len(fd_scale)
    # _logger.debug(ind_str + 'N_points * N_events = %i * %i' % (n_points, n_events))

    fd_env = np.exp(-(fd_scale - spec_center) ** 2 / 2 / spec_width_sigm ** 2)
    td_scale = np.linspace(0, 2 * np.pi / (fd_scale[1] - fd_scale[0]) * hr_eV_s * speed_of_light, n_points)

    if flattop:
        td_env = np.zeros_like(td_scale)
        il = find_nearest_idx(td_scale, np.mean(td_scale) - pulse_length * 1e-6 / 2)
        ir = find_nearest_idx(td_scale, np.mean(td_scale) + pulse_length * 1e-6 / 2)
        td_env[il:ir] = 1
    else:
        s0 = np.mean(td_scale)
        td_env = np.exp(-(td_scale - s0) ** 2 / 2 / (pulse_length_sigm * 1e-6) ** 2)

    result = imitate_1d_sase_like(td_scale, td_env, fd_scale, fd_env, phen0=spec_center, en_pulse=en_pulse,
                                  fit_scale='fd', n_events=n_events, **kwargs)

    return result