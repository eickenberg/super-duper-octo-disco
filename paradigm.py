# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict


def check_stim_durations(stim_onsets, stimDurations):
    """ If no durations specified (stimDurations is None or empty np.array)
    then assume spiked stimuli: return a sequence of zeros with same
    shape as onsets sequence.
    Check that durations have same shape as onsets.

    """
    nbc = len(stim_onsets)
    nbs = len(stim_onsets[stim_onsets.keys()[0]])
    if (stimDurations is None or
        (type(stimDurations) == list and
         all([d is None for d in stimDurations]))):
        dur_seq = [[np.array([]) for s in xrange(nbs)] for i in xrange(nbc)]
        stimDurations = OrderedDict(zip(stim_onsets.keys(), dur_seq))

    if stimDurations.keys() != stim_onsets.keys():
        raise Exception('Conditions in stimDurations (%s) differ '
                        'from stim_onsets (%s)' % (stimDurations.keys(),
                                                   stim_onsets.keys()))

    for cn, sdur in stimDurations.iteritems():
        for i, dur in enumerate(sdur):
            if dur is None:
                stimDurations[cn][i] = np.zeros_like(stim_onsets[cn][i])
            elif hasattr(dur, 'len') and len(dur) == 0:
                stimDurations[cn][i] = np.zeros_like(stim_onsets[cn][i])
            elif hasattr(dur, 'size') and dur.size == 0:
                stimDurations[cn][i] = np.zeros_like(stim_onsets[cn][i])
            else:
                if not isinstance(stimDurations, np.ndarray):
                    stimDurations[cn][i] = np.array(dur)
                assert len(stimDurations[cn][i]) == len(stim_onsets[cn][i])

    return stimDurations


def extend_sampled_events(sampled_events, sampled_durations):
    """ Add events to encode stimulus duration
    """
    extended_events = set(sampled_events)
    for io, o in enumerate(sampled_events):
        extended_events.update(range(o + 1, o + sampled_durations[io]))

    return np.array(sorted(list(extended_events)), dtype=int)


def restarize_events(events, durations, dt, t_max):
    """ build a binary sequence of events. Each event start is approximated
    to the nearest time point on the time grid defined by dt and t_max.
    """
    smpl_events = np.array(np.round_(np.divide(events, dt)), dtype=int)
    smpl_durations = np.array(np.round_(np.divide(durations, dt)), dtype=int)
    smpl_events = extend_sampled_events(smpl_events, smpl_durations)
    if np.allclose(t_max % dt, 0):
        bin_seq = np.zeros(int(t_max / dt) + 1)
    else:
        bin_seq = np.zeros(int(np.round((t_max + dt) / dt)))
    bin_seq[smpl_events] = 1

    return bin_seq


class Paradigm:
    def __init__(self, stimOnsets, sessionDurations=None, stimDurations=None):
        """
        Args:
            *stimOnsets* (dict of list) :
                dictionary mapping a condition name to a list of session
                stimulus time arrivals.
                eg:
                {'cond1' : [<session 1 onsets>, <session 2 onsets>]
                 'cond2' : [<session 1 onsets>, <session 2 onsets>]
                 }
            *sessionDurations* (1D numpy float array): durations for all sessions
            *stimDurations* (dict of list) : same structure as stimOnsets.
                             If None, spiked stimuli are assumed (ie duration=0).
        """
        self.stimOnsets = stimOnsets
        self.stimDurations = check_stim_durations(stimOnsets, stimDurations)
        self.nbSessions = len(self.stimOnsets[self.stimOnsets.keys()[0]])
        self.sessionDurations = sessionDurations

    def get_stimulus_names(self):
        return self.stimOnsets.keys()

    def get_t_max(self):
        ns = len(self.sessionDurations)
        return max([self.sessionDurations[i] for i in xrange(ns)])

    def get_rastered(self, dt, tMax=None):
        """ Return binary sequences of stimulus arrivals. Each stimulus event
        is approximated to the closest time point on the time grid defined
        by dt. eg return:
        { 'cond1' : [np.array([ 0 0 0 1 0 0 1 1 1 0 1]),
                     np.array([ 0 1 1 1 0 0 1 0 1 0 0])] },
          'cond2' : [np.array([ 0 0 0 1 0 0 1 1 1 0 0]),
                     np.array([ 1 1 0 1 0 1 0 0 0 0 0])] },

        Arg:
            - dt (float): temporal resolution of the target grid
            - tMax (float): total duration of the paradigm
                            If None, then use the session lengths
        """
        rasteredParadigm = OrderedDict({})
        if tMax is None:
            tMax = self.get_t_max()
        for cn in self.stimOnsets.iterkeys():
            par = []
            for iSess, ons in enumerate(self.stimOnsets[cn]):
                dur = self.stimDurations[cn][iSess]
                binaryEvents = restarize_events(ons, dur, dt, tMax)
                par.append(binaryEvents)
            rasteredParadigm[cn] = np.vstack(par)

        return rasteredParadigm

