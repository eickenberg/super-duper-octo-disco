import numpy as np

from paradigm import Paradigm

PHY_PARAMS_FRISTON00 = {
    'model_name': 'Friston00',
    'tau_s': 1 / .8,
    'eps': .5,
    'eps_max': 10.,
    'tau_m': 1.,
    'tau_f': 1 / .4,
    'alpha_w': .2,
    'E0': .8,
    'V0': .02,
    'TE': 0.04,
    'r0': 100,  # 25 at 1.5T, and rO = 25 (B0/1.5)**2
    'vt0': 80.6,  # 40.3 at 1.5T, and vt0 = 40.3 (B0/1.5)
    'e': 0.4,
    'model': 'RBM',
    'linear': True,
    'obata': False,
    'buxton': False
}

PHY_PARAMS_KHALIDOV11 = {
    'model_name': 'Khalidov11',
    'tau_s': 1.54,
    'eps': .54,
    'eps_max': 10.,  # TODO: check this
    'tau_m': 0.98,
    'tau_f': 2.46,
    'alpha_w': .33,
    'E0': .34,
    'V0': 1,   # obata 0.04, Friston and others 0.02, Griffeth13 0.05
    'r0': 100,  # 25 at 1.5T, and rO = 25 (B0/1.5)**2
    'vt0': 80.6,  # 40.3 at 1.5T, and vt0 = 40.3 (B0/1.5)
    'e': 1.43,  # 0.4 or 1
    'TE': 0.018,
    'model': 'RBM',
    'linear': False,
    'obata': False,
    'buxton': False
}


def phy_integrate_euler(phy_params, tstep, stim, epsilon, Y0=None):
    """
    Integrate the ODFs of the physiological model with the Euler method.

    Args:
        - phy_params (dict (<param_name> : <param_value>):
            parameters of the physiological model
        - tstep (float): time step of the integration, in seconds.
        - stim (np.array(nb_steps, float)): stimulation sequence with temporal
            resolution equal to the time step of the integration
        - epsilon (float): neural efficacy
        - Y0 (np.array(4, float) | None): initial values for the physiological
                                          signals.
                                          If None: [0, 1,   1, 1.]
                                                    s  f_in q  v
    Result:
        - np.array((4, nb_steps), float)
          -> the integrated physiological signals, where indexes of the first
          axis correspond to:
              0 : flow inducing
              1 : inflow
              2 : HbR
              3 : blood volume

    TODO: should the output signals be rescaled wrt their value at rest?
    """

    epsilon = phy_params['eps']  # WARNING!! Added to compute figures
    tau_s = phy_params['tau_s']
    tau_f = phy_params['tau_f']
    tau_m = phy_params['tau_m']
    alpha_w = phy_params['alpha_w']
    E0 = phy_params['E0']

    def cpt_phy_model_deriv(y, s, epsi, dest):
        N, f_in, v, q = y
        if f_in < 0.:
            #raise Exception('Negative f_in (%f) at t=%f' %(f_in, ti))
            # HACK
            print 'Warning: Negative f_in (%f) at t=%f' % (f_in, ti)
            f_in = 1e-4

        dest[0] = epsi * s - (N / tau_s) - ((f_in - 1) / tau_f)  # dNdt
        dest[1] = N  # dfidt
        dest[2] = (1 / tau_m) * (f_in - v**(1 / alpha_w))  # dvdt
        dest[3] = (1 / tau_m) * ((f_in / E0) * (1 - (1 - E0)**(1 / f_in)) -
                                 (q / v) * (v**(1 / alpha_w)))  # dqdt
        return dest

    res = np.zeros((stim.size + 1, 4))
    res[0, :] = Y0 or np.array([0., 1., 1., 1.])
    #res[0, :] = Y0 or np.array([0., 0., 0., 0.])

    for ti in xrange(1, stim.size + 1):
        cpt_phy_model_deriv(res[ti - 1], stim[ti - 1], epsilon, dest=res[ti])
        res[ti] *= tstep
        res[ti] += res[ti - 1]

    return res[1:, :].T


def create_evoked_physio_signals(physiological_params, paradigm,
                                 neural_efficacies, dt, integration_step=.05):
    """
    Generate evoked hemodynamics signals by integrating a physiological model.

    Args:
        - physiological_params (dict (<pname (str)> : <pvalue (float)>)):
             parameters of the physiological model.
             In jde.sandbox.physio see PHY_PARAMS_FRISTON00, PHY_PARAMS_FMRII
        - paradigm (pyhrf.paradigm.Paradigm) :
             the experimental paradigm
        - neural_efficacies (np.ndarray (nb_conditions, nb_voxels, float)):
             neural efficacies involved in flow inducing signal.
        - dt (float):
             temporal resolution of the output signals, in second
        - integration_step (float):
             time step used for integration, in second

    Returns:
        - np.array((nb_signals, nb_scans, nb_voxels), float)
          -> All generated signals, indexes of the first axis correspond to:
              - 0: flow inducing
              - 1: inflow
              - 2: blood volume
              - 3: [HbR]
    """
    # TODO: handle multiple conditions
    # -> create input activity signal [0, 0, eff_c1, eff_c1, 0, 0, eff_c2, ...]
    # for now, take only first condition
    first_cond = paradigm.get_stimulus_names()[0]
    stim = paradigm.get_rastered(integration_step)[first_cond][0]
    neural_efficacies = neural_efficacies[0]

    # response matrix intialization
    integrated_vars = np.zeros((4, neural_efficacies.shape[0], stim.shape[0]))
    for i, epsilon in enumerate(neural_efficacies):
        integrated_vars[:, i, :] = phy_integrate_euler(physiological_params,
                                                       integration_step, stim,
                                                       epsilon)
    # downsampling:
    nb_scans = paradigm.get_rastered(dt)[first_cond][0].size
    dsf = int(dt / integration_step)
    return np.swapaxes(integrated_vars[:, :, ::dsf][:, :, :nb_scans], 1, 2)


def create_k_parameters(physiological_params):
    """ Create field strength dependent parameters k1, k2, k3
    """
    # physiological parameters
    V0 = physiological_params['V0']
    E0 = physiological_params['E0']
    TE = physiological_params['TE']
    r0 = physiological_params['r0']
    vt0 = physiological_params['vt0']
    e = physiological_params['e']
    if physiological_params['model'] == 'RBM':  # RBM
        k1 = 4.3 * vt0 * E0 * TE
        k2 = e * r0 * E0 * TE
        k3 = 1 - physiological_params['e']
    elif physiological_params['buxton']:
        k1 = 7 * E0
        k2 = 2
        k3 = 2 * E0 - 0.2
    else:   # CBM
        k1 = (1 - V0) * 4.3 * vt0 * E0 * TE
        k2 = 2. * E0
        k3 = 1 - physiological_params['e']
    physiological_params['k1'] = k1
    physiological_params['k2'] = k2
    physiological_params['k3'] = k3

    return k1, k2, k3


def create_bold_from_hbr_and_cbv(physiological_params, hbr, cbv):
    """
    Compute BOLD signal from HbR and blood volume variations obtained
    by a physiological model
    """

    # physiological parameters
    V0 = physiological_params['V0']
    k1, k2, k3 = create_k_parameters(physiological_params)

    # linear vs non-linear
    if physiological_params['linear']:
        sign = 1.
        if physiological_params['obata']:  # change of sign
            sign = -1
        bold = V0 * ((k1 + k2) * (1 - hbr) + sign * (k3 - k2) * (1 - cbv))
    else:  # non-linear
        bold = V0 * (k1 * (1 - hbr) + k2 * (1 - hbr / cbv) + k3 * (1 - cbv))

    return bold


def physio_hrf(hrf_length=25., dt=.5, normalize=False,
               physiological_params=PHY_PARAMS_KHALIDOV11):
    """
    Generate a BOLD response function by integrating a physiological model and
    setting its driving input signal to a single impulse.

    Args:
        - physiological_params (dict (<pname (str)> : <pvalue (float)>)):
            parameters of the physiological model.
            In jde.sandbox.physio see PHY_PARAMS_FRISTON00, PHY_PARAMS_FMRII...
        - response_dt (float): temporal resolution of the response, in second
        - response_duration (float): duration of the response, in second

    Return:
        - np.array(nb_time_coeffs, float)
          -> the BRF (normalized)
        - also return brf_not_normalized, q, v when return_prf_q_v=True
          (for error checking of v and q generation in calc_hrfs)
    """
    # Paradigm(onsets, sessionduration, stimduration)
    p = Paradigm({'c': [np.array([0.])]}, [hrf_length],
                 {'c': [np.array([1.])]})
    n = np.array([[1.]])
    s, f, v, q = create_evoked_physio_signals(physiological_params, p, n, dt)
    brf = create_bold_from_hbr_and_cbv(physiological_params, q[:, 0], v[:, 0])
    if normalize:
        return brf / (brf**2).sum()**.5
    else:
        return brf


def gen_hrf_bezier_can(hrf_length=25., dt=.5, normalize=False):
    """Generates canonical HRF using bezier curves"""
    tPic = 5
    if tPic >= duration:
        tPic = duration * .3
    tus = np.round(tPic + (duration - tPic) * 0.5)
    h = bezier_hrf(hrf_length=25., dt=.5, pic=[tPic,1],
                       ushoot=[tus, -0.2], normalize=normalize)
    return h


def bezier_hrf(hrf_length=25., dt=.5, pic=[6,1], picw=2,
               ushoot=[15,-0.2], ushootw=3, normalize=False):
    """Generates an HRF using bezier curves"""

    timeAxis = np.arange(0, hrf_length, dt)
    prec = (timeAxis[1] - timeAxis[0]) / 20.

    partToPic = bezier_curve([0., 0.], [2., 0.], pic, [pic[0] - picw, pic[1]], prec)
    partToUShoot = bezier_curve(pic, [pic[0] + picw, pic[1]], ushoot,
                               [ushoot[0] - ushootw, ushoot[1]], prec)
    partToEnd = bezier_curve(ushoot, [ushoot[0] + ushootw, ushoot[1]],
                            [timeAxis[-1], 0.], [timeAxis[-1] - 1, 0.], prec)
    hrf = range(2)
    hrf[0] = partToPic[0] + partToUShoot[0] + partToEnd[0]
    hrf[1] = partToPic[1] + partToUShoot[1] + partToEnd[1]

    # check if bezier parameters are well set to generate an injective curve
    assert (np.diff(np.array(hrf[0])) >= 0).all()

    # Resample time axis
    iSrc = 0
    resampledHRF = []
    for itrgT in xrange(0, len(timeAxis)):
        t = timeAxis[itrgT]
        while (iSrc+1 < len(hrf[0])) and (np.abs(t-hrf[0][iSrc]) > np.abs(t-hrf[0][iSrc+1])):
            iSrc += 1
        resampledHRF.append(hrf[1][iSrc])
        iSrc += 1
    hvals = np.asarray(resampledHRF)

    if normalize:
        resampledHRF = hvals / (hvals**2).sum()**.5

    return hvals


def bezier_curve(p1, pc1, p2, pc2, xPrecision):
    """Creates Bezier curve given some points and control points"""
    bezierCtrlPoints = []
    bezierPoints = []
    bezierCtrlPoints.append(pc1)
    bezierCtrlPoints.append(pc2)
    bezierPoints.append(p1)
    bezierPoints.append(p2)

    precisionCriterion = 0
    nbPoints = 1

    while( precisionCriterion != nbPoints):
        precisionCriterion = 0
        nbPoints = len(bezierPoints)-1
        ip = 0
        ipC = 0
        nbP = 0

        while(nbP<nbPoints) :
            e1 = bezierPoints[ip]
            e2 = bezierPoints[ip+1]
            c1 = bezierCtrlPoints[ipC]
            c2 = bezierCtrlPoints[ipC+1]

            div2 = [2.,2.]
            m1 = np.divide(np.add(c1,e1),div2)
            m2 = np.divide(np.add(c2,e2),div2)
            m3 = np.divide(np.add(c1,c2),div2)
            m4 = np.divide(np.add(m1,m3),div2)
            m5 = np.divide(np.add(m2,m3),div2)
            m = np.divide(np.add(m4,m5),div2)

            bezierCtrlPoints[ipC] = m1
            bezierCtrlPoints.insert(ipC+1, m5)
            bezierCtrlPoints.insert(ipC+1, m4)
            bezierCtrlPoints[ipC+3] = m2

            bezierPoints.insert(ip+1, m)

            # Stop criterion :
            if abs(m[0]-bezierPoints[ip][0]) < xPrecision :
                precisionCriterion += 1

            nbP += 1
            ip += 2
            ipC += 4

    return ([p[0] for p in bezierPoints],[p[1] for p in bezierPoints])



