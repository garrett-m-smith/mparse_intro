#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:27:19 2020

@author: garrettsmith

Tested w/ Python 3.7.6
"""

from string import punctuation, digits
from itertools import product, combinations
import numpy as np
from scipy.linalg import eig as eiglr
from scipy.linalg import expm
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd


# A head is only allowed on of each of these.
ONE_ALLOWED = ['det', 'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'obj',
               'iobj', 'ccomp', 'xcomp']


class Word:
    def __init__(self, word, token_nr):
        self.word: str = word
        self.word_nr: int = token_nr
        self.token: str = self.word + str(self.word_nr)
        #self.token: str = self.word + str(self.word_nr) + self.pos

    def __eq__(self, o):
        return self.token == o.token

    def __repr__(self):
        return self.word

    def __hash__(self):
        return hash(repr(self))


class Sentence:
    def __init__(self, sent):
        self.sentence: str = sent
        #words = sent.strip(punctuation).split(' ')
        words = sent.split(' ')
        self.words = []
        for i, w in enumerate(words):
            self.words.append(Word(w, i))

    def __eq__(self, o):
        return all(self.words == o.words)

    def __repr__(self):
        return self.sentence


class Link:
    def __init__(self, gov, rel, dep, order=0, harmony=0.0):
        self.gov: Word = gov
        self.dep: Word = dep
        self.rel: str = rel
        # Convention: sgn(dep pos. - gov pos.)
        # So, -1 means the gov follows the dep., +1 means gov precedes dep
        # 0 is unspecified
        self.order = order
        self.harmony = harmony

    def __repr__(self):
        return ''.join([self.rel, '(', self.gov.token, ', ', self.dep.token, ')'])

    def __eq__(self, o):
        return self.gov == o.gov and self.dep == o.dep and self.rel == o.rel
    
    def __lt__(self, o):
        # "Less than" for sorting
        return self.__repr__() < o.__repr__()

    def __hash__(self):
        return hash(repr(self))


class Configuration:
    """Basically a glorified list of links with an optional harmony
    """
    def __init__(self, links=None, harmony=np.nan):
        self.links = []
        self.links.extend(links)
        self.harmony = harmony
        self._len = len(self.links)

    def __repr__(self):
        return ('\n'.join(map(str, self.links)))

    def __eq__(self, o):
        return (self.__class__ == o.__class__ and
                self._len == o._len and
                all(i == j for i, j in zip(self.links, o.links)))
    def __lt__(self, o):
        # "Less than" for sorting
        return self.__repr__() < o.__repr__()

    def __len__(self):
        if self.links == [Link(Word('X', 0), 'X', Word('X', 0))]:
            return 0
        else:
            return self._len

    def __hash__(self):
        return(hash(repr(self)))

    def add_link(self, link):
        assert isinstance(link, Link), "Attempted to add something that wasn't a link to a configuration"
        self.links.append(link)

    def set_harmony(self, harmony):
        self.harmony = harmony


class Segment:
    """The principle unit for mparse.
    """
    def __init__(self, sent_so_far=None, grammar=None):
        self.sent_so_far = sent_so_far
        self.words_so_far = sent_so_far.words
        self.links = []
        self.configs = []
        self.grammar = grammar
        self.rels = list(set([x.split('-')[0] for x in self.grammar]))
        self.adjs = []

    def _make_links(self):
        """Brute force enumeration of all possible links given the words in the
        sentence and the relations in the grammar.
        """
        for x in product(self.words_so_far, self.rels, self.words_so_far):
            relgov = '-'.join([str(x[1]), str(x[0])])
            if (self.grammar.get(relgov)
                and self.grammar[relgov].get(str(x[2]))
                and x[0].token != x[2].token):
                self.links.append(Link(gov=x[0], rel=x[1], dep=x[2]))
        # Subsetting just the unique links
        seen = set()
        unq = []
        for x in self.links:
            if x not in seen:
                unq.append(x)
                seen.add(x)
        self.links = unq

    def _init_configs(self, incompat=None):
        """Takes a list of dependent words and a list of allowed
        governor-relation links. Builds a list of configurations.
        """
        # Creating links between attch. sites and words
        self.configs = []
        for i in range(len(self.words_so_far) - 1):
            for comb in combinations(self.links, i + 1):
                # Make sure each core gov-rel appears only once
                if not len(set([(y.rel, y.gov) for y in comb if y.rel in ONE_ALLOWED])) == len([(y.rel, y.gov) for y in comb if y.rel in ONE_ALLOWED]):
                    continue
                # Make sure each word only has one gov
                elif not len(set([y.dep for y in comb])) == len([y.dep for y in comb]):
                #if not len(set([y.dep for y in comb])) == len([y.dep for y in comb]):
                    continue
                else:
                    self.configs.append(Configuration(links=[Link(y.gov, y.rel, y.dep) for y in comb]))
        # Removing given pairs. This is a kludge to get around dealing with
        # lexical ambiguity.
        torm = []
        if incompat:
            for pair in incompat:
                rel0, gov0 = pair[0].split('-')
                rel1, gov1 = pair[1].split('-')
                for i, c in enumerate(self.configs):
                    match0 = [l for l in c.links if str(l.rel) == rel0 and str(l.gov) == gov0]
                    assert len(match0) <= 1, 'too many matches'
                    match1 = [l for l in c.links if str(l.rel) == rel1 and str(l.gov) == gov1]
                    assert len(match1) <= 1, 'too many matches'
                    if match0 and match1 and match0[0] in c.links and match1[0] in c.links:
                        torm.append(i)
            #print('rming: {}'.format(torm))
            self.configs = [c for j, c in enumerate(self.configs) if j not in torm]
        self._make_dep_adj_matrices()
        # Make sure that there are no cycles and no deps w/ multiple heads
        # Also double check to make sure that all of the links in the configs
        # are in the grammar.
        self.configs = [self.configs[i] for i, x in enumerate(self.adjs) if not
                        (self._contains_cycle(x) or self._is_multi_gov(x))
                        and all(self.grammar.get('-'.join([y.rel, str(y.gov)])).get(str(y.dep))
                                for y in self.configs[i].links)]
        self.configs.insert(0, Configuration([Link(Word('X', 0), 'X', Word('X', 0))]))
        # Create a list of absorbing state indices
        configlengths = [len(c) for c in self.configs]
        self.absorbing = [i for i, c in enumerate(configlengths) if c == max(configlengths)]


    def _calc_harmony(self, config):
        """Takes a configuration and a nested dictionary of link harmonies
        and returns the harmony of the configuration.

        Open question: whether to use the avg. harmony or the sum of link
        harmonies.
        """
        gr = self.grammar
        nwords = len(self.words_so_far)
        if config == Configuration(links=[Link(Word('X', 0), 'X', Word('X', 0))]):
            harmony = -nwords + 1
        else:
            h = []
            dists = []
            orders = []
            for l in config.links:
                try:
                    h.append(gr['-'.join([l.rel, l.gov.word])][l.dep.word][0])
                    dists.append(abs(l.gov.word_nr - l.dep.word_nr))
                    orders.append(gr['-'.join([l.rel, l.gov.word])][l.dep.word][1] * np.sign(l.dep.word_nr - l.gov.word_nr))
                except KeyError:
                    print('Ah! No grammar for {}'.format(l))
            harmony = -abs(len(config) - nwords + 1) + sum(h) - sum(dists) + sum(orders)#/len(h)
        config.set_harmony(harmony)

    def _make_dep_adj_matrices(self):
        """Creates an adjacency matrix where element adj[i,j] is one if the
        i-th word is the governor of the j-th word and zero otherwise.
        """
        # Elements: [gov, dep]
        for config in self.configs:
            links = [(x.gov, x.dep) for x in config.links]
            # For indexing the dimensions
            idx = {x: i for i, x in enumerate(self.words_so_far)}
            curradj = np.zeros((len(self.words_so_far), len(self.words_so_far)))
            for l in links:
                curradj[idx[l[0]], idx[l[1]]] += 1
            self.adjs.append(curradj)

    def _contains_cycle(self, curradj):
        """Takes an adjacency matrix and returns True if it contains a
        cycle. Based on the algorithm in Norman (1965). AIChE Journal 11(3),
        with the efficient implementation of matrix powers from:
        https://stackoverflow.com/questions/16436165/detecting-cycles-in-an-adjacency-matrix"""
        n = curradj.shape[0]  # adjacency matrix will be square
        An = curradj.copy()
        for _ in range(2, n + 1):
            An = An.dot(curradj)  # do not re-compute A^n from scratch
            # If there are cycles, the diagonal of the matrix will not
            # be zero
            if np.trace(An) != 0:
                return True
            else:
                return False

    def _is_multi_gov(self, curradj):
        """Takes an adjacency matrix and returns True if any dependent has
        more than one governor.
        """
        return any(curradj.sum(axis=0) > 1)

    def _arrhenius(self, hold, hnew, T):
        if hold == -np.inf:
            return np.exp(hnew / T)  # or should we use hold?
        else:
            return np.exp((hnew - hold) / T)

    def _metropolis(self, hold, hnew, T):
        return np.minimum(1.0, np.exp((hnew - hold) / T))

    def _glauber(self, hold, hnew, T):
        return 1. / (1 + np.exp(-(hnew - hold) / T))

    def _assign_sets(self, c1, c2):
        """Utility function for setting up configs as sets for creating the
        transition matrix.
        """
        nolinks = Configuration([Link(Word('X', 0), 'X', Word('X', 0))])
        if (c1 == nolinks) and (c2 != nolinks):
            x = set()
            y = set(c2.links)
        elif (c1 != nolinks) and (c2 == nolinks):
            x = set(c1.links)
            y = set()
        elif c1 == c2 == nolinks:
            x = y = set()
        else:
            x = set(c1.links)
            y = set(c2.links)
        return x, y

    def _basic_transition_matrix(self):
        """Assigns dimensions in order given in configs.
        """
        # For the first word, probability flows from a dummy config
        if self.configs == [Configuration([Link(Word('X', 0), 'X', Word('X', 0))])]:
            self.base_trans_mat = np.array([[0., 0], [1, 0]])
        else:
            ndim = len(self.configs)
            W = np.zeros((ndim, ndim))
            for i, j in combinations(range(ndim), 2):
                x, y = self._assign_sets(self.configs[i], self.configs[j])
                if x < y and len(y.difference(x)) == 1:
                    # If the longer config has the max. pos. links given the
                    # words, don't allow prob. to flow away from it.
                    #if len(y) != len(self.words_so_far) - 1:
                    if len(y) != max(map(len, self.configs)):
                        W[i, j] = W[j, i] = 1
                    else:
                        W[j, i] = 1
                        W[:, j] = 0
                else:
                    continue
            self.base_trans_mat = W

    def plot_transition_matrix(self, basic=False):
        plt.figure()
        if basic:
            plt.imshow(self.base_trans_mat)
        else:
            plt.imshow(self.trans_mat)
        if len(self.configs) == 1:
            labs = ['Dummy', self.configs[0]]
            tcs = range(2)
        else:
            labs = self.configs
            tcs = range(len(self.configs))
        plt.colorbar()
        plt.xticks(ticks=tcs, labels=labs,
                   rotation=45, horizontalalignment="right")
        plt.yticks(ticks=tcs, labels=labs)
        plt.show()

    def make_transition_matrix(self, tau, T, method):
        """Takes a scaling parameter (tau) a noise parameter (T), and
        a method for calculating the transition rates.
        """
        Wij = self.base_trans_mat.copy()
        for (i, j) in np.argwhere(Wij):
            if method.startswith('m'):
                if len(self.configs) == 1:
                    Wij[i, j] = self._metropolis(-np.inf,
                                                 self.configs[0].harmony, T)
                else:
                    Wij[i, j] = self._metropolis(self.configs[j].harmony,
                                                 self.configs[i].harmony, T)
            elif method.startswith('g'):
                if len(self.configs) == 1:
                    Wij[i, j] = self._glauber(-np.inf,
                                                 self.configs[0].harmony, T)
                else:
                    Wij[i, j] = self._glauber(self.configs[j].harmony,
                                              self.configs[i].harmony, T)
            elif method.startswith('a'):
                if len(self.configs) == 1:
                    Wij[i, j] = self._arrhenius(-np.inf,
                                                 self.configs[0].harmony, T)
                else:
                    Wij[i, j] = self._arrhenius(self.configs[j].harmony,
                                                self.configs[i].harmony, T)
        Wij *= tau * Wij.shape[0]  # transition rate *per* config
        np.fill_diagonal(Wij, 0.0)
        # Checking assumptions of W-matrices, van Kampen p. 101
        assert np.all(Wij >= 0), 'Non-positive matrix entries'
        # # Ensuring the columns sum to zero:
        np.fill_diagonal(Wij, -1 * Wij.sum(axis=0))
        assert np.allclose(Wij.sum(axis=0), 0), 'Columns do not sum to 0'
        self.trans_mat = Wij + 0.0

    def make_fp_matrices(self):
        """Generate matrices A (within-bounds dynamics) and  B (dynamics of
        probability moving into the absorbing states)
        """
        if len(self.configs) == 1:
            inbounds = [0]
            absorbing = [1]
        else:
            inbounds = list(range(len(self.configs)))
            inbounds = [i for i in inbounds if i not in self.absorbing]
            absorbing = self.absorbing
        self.inbounds = inbounds
        self.Amat = self.trans_mat[np.ix_(inbounds, inbounds)]
        self.Bmat = self.trans_mat[np.ix_(absorbing, inbounds)]
        #print(self.Amat, self.Bmat)

    def check_p0(self, P0):
        self.origP0 = P0
        P0 = P0[self.inbounds]
        if len(self.configs) == 1:
            P0 = np.array([1.0])
        else:
            P0 = P0[self.inbounds]
        if np.all(P0 == 0):
            #print('No new structures have been added; using uniform distribution as initial condition.')
            P0 = np.ones(len(self.inbounds))
            P0 /= P0.sum()
        self.P0 = P0
        return P0

    def first_exit_moments(self, P0):
        """First exit time moments, Oppenheim (1977, p. 89). How long
        does it take to reach a configuration of the correct number of
        links?
        """
        U = np.ones(self.Amat.shape[0])
        moments = []
        for n in range(1, 3):
            moments.append((-1)**n * np.math.factorial(n) *
                            U.dot(np.linalg.matrix_power(self.Amat, -n).dot(P0)))
        self.mean = moments[0]
        self.var = moments[1] - moments[0]**2

    def splitting_probabilities(self, P0):
        """Splitting probability  (Valleriani & Kolomiesky, 2014)
        The i-th entry gives the probability of absorbtion in the i-th
        absorbing state given that you started with the initial probability
        distribution P0.
        """
        self.spl_probs = -self.Bmat.dot(np.linalg.inv(self.Amat).dot(P0))
        #print(P0)
        assert np.isclose(self.spl_probs.sum(), 1.0), "Splitting probabilities don't sum to one."
        #print('\t', [(c, s) for c, s in zip([self.configs[i] for i in self.absorbing], self.spl_probs)], sep='')

    def plot_fptd(self, P0, overall=True, check=False, log=False):
        mn = self.mean
        dt = mn / 100.0
        #trange = np.arange(0, 3 * mn, dt)
        trange = np.geomspace(0.01, 3*mn, 100)
        if overall:
            dens = lambda t: self.Bmat.dot(expm(self.Amat * t).dot(P0[self.inbounds])).sum()
            if check:
                assert np.isclose(1.0, quad(dens, 0, np.inf)[0]), "Exit time density doesn't sum to one"
            plt.plot(trange, [dens(t) for t in trange])
            plt.title('{}: Exit time distribution'.format(self.words_so_far))
            plt.xlabel('Time (arbitrary units)')
            if log:
                plt.xscale('log')
            plt.show()
        else:
            dens = lambda t: self.Bmat.dot(expm(self.Amat * t).dot(P0[self.inbounds])) / self.spl_probs  # Oppenheim normalized
            plt.plot(trange, [dens(t) for t in trange])#, label=self.configs[idx[i]])
            plt.legend([self.configs[i] for i in self.absorbing])
            plt.title('{}: First passage time distributions'.format(self.words_so_far))
            plt.xlabel('Time (arbitrary units)')
            if log:
                plt.xscale('log')
            plt.show()

    def prep_segment(self, incompat=None, tau=1.0, T=0.5, method='glauber'):
        self._make_links()
        self._init_configs(incompat=incompat)
        for c in self.configs:
            self._calc_harmony(c)
            #c.links = sorted(c.links)
        #self.configs = sorted(self.configs)
        self._basic_transition_matrix()
        self.make_transition_matrix(tau, T, method)
        self.make_fp_matrices()

    def plot_seg_soln(self):
        tsteps = 500
        tvec = np.geomspace(0.001, 0.01*self.mean, tsteps)
        soln = np.zeros((tsteps, len(self.inbounds)))
        tmpP0 = np.ones(len(self.configs))
        tmpP0 /= tmpP0.sum()
        for t in range(tsteps):
            #soln[t,:] = expm(self.trans_mat * t).dot(self.origP0)
            soln[t,:] = expm(self.Amat * t).dot(self.P0)
        for i in range(len(self.inbounds)):
            plt.plot(tvec, soln[:,i], label=self.configs[self.inbounds[i]])
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()


def run_sent(sent, grammar, incompat, tau=1.0, T=0.5, method='glauber',
             plot_trans=False, plot_fptd=False, check_fptd=False,
             exit_only=True, log_axis=False, verbose=True):
    segs = []
    for w in range(len(sent)):
        if verbose:
            print(sent[:w+1])
        segs.append(Segment(Sentence(' '.join(sent[:w+1])), grammar))
        segs[w].prep_segment(incompat, tau, T, method)
        if plot_trans:
            segs[w].plot_transition_matrix(basic=False)
        if w == 0:
            P0 = np.zeros(len(segs[w].configs))
            P0[segs[w].inbounds] = 1.0
            P0 /= P0.sum()
        else:
            P0 = np.zeros(len(segs[w].configs))
            # Get the absorbing states of the previous segment
            prevabs = [segs[w-1].configs[i] for i in segs[w-1].absorbing]
            # Use the prev. splitting probabilities as new init. cond.
            for i, p in enumerate(prevabs):
                for j, c in enumerate(segs[w].configs):
                    if c == p:
                        P0[j] = segs[w-1].spl_probs[i]
        # Check to make sure that there's no probability in the
        # current absorbing states
        P0 = segs[w].check_p0(P0)
        segs[w].P0 = P0
        segs[w].first_exit_moments(P0)
        if verbose:
            print('\tMean: {}, Var: {}'.format(np.round(segs[w].mean, 3), np.round(segs[w].var, 3)))
        segs[w].splitting_probabilities(P0)
        if plot_fptd:
            segs[w].plot_fptd(P0, check=check_fptd, overall=exit_only, log=log_axis)
    return segs


def explore_params(sent, grammar, incompat, taulist, Tlist,
                   method='glauber'):
    dat = []
    for currtau in taulist:
        for currT in Tlist:
            segs = []
            segdat = []
            for w in range(len(sent)):
                #print(sent[:w+1])
                segs.append(Segment(Sentence(' '.join(sent[:w+1])), grammar))
                segs[w].prep_segment(incompat, currtau, currT, method)
                if w == 0:
                    P0 = np.zeros(len(segs[w].configs))
                    P0[segs[w].inbounds] = 1.0
                    P0 /= P0.sum()
                else:
                    P0 = np.zeros(len(segs[w].configs))
                    prevabs = [segs[w-1].configs[i] for i in segs[w-1].absorbing]
                    for i, p in enumerate(prevabs):
                        for j, c in enumerate(segs[w].configs):
                            if c == p:
                                P0[j] = segs[w-1].spl_probs[i]
                    P0 = segs[w].check_p0(P0)
                segs[w].first_exit_moments(P0)
                segs[w].splitting_probabilities(P0)
                segdat.append({'seg_nr': w,
                               'segment': ' '.join(map(str, segs[w].words_so_far)),
                               'sent': ' '.join(sent),
                               'tau': currtau,
                               'T': currT,
                               'mean': segs[w].mean,
                               'variance': segs[w].var,
                               'splits': segs[w].spl_probs})
            dat.extend(segdat)
    return pd.DataFrame(dat)


# def solve(self, dt=0.1):
#     print('consider rewriting to have initial condition at first word only have probability in the length-one configurations')
#     times = [int(2 * x.mean / dt) for x in self.segments]
#     #times = [int(x.mean / dt) for x in self.segments]
#     tends = np.cumsum(times)
#     tmax = sum([2 * x for x in times])
#     full = np.ones((1, len(self.configs))) / len(self.configs)
#     for i, seg in enumerate(self.segments):
#         if seg.mean != 0:
#             traj = np.zeros((int(times[i] / dt), len(self.configs)))
#         else:
#             print("You're in a kludge! Deal with zero RTs!!!")
#             traj = np.zeros((int(max(times) / dt), len(self.configs)))
#         traj[0, :] = full[-1, :]
#         for t in range(traj.shape[0] - 1):
#             traj[t + 1, :] = traj[t, :] + dt*seg.trans_mat.dot(traj[t, :])
#         full = np.row_stack([full, traj])
#     self.soln = full
#     self.tvec = np.linspace(0, sum([2 * x.mean for x in self.segments]), full.shape[0])

# def plot_soln(self):
#     plt.figure(figsize=(10, 6))
#     lineobjs = plt.plot(self.tvec, self.soln)
#     plt.legend(lineobjs, self.configs, loc='upper left',
#         bbox_to_anchor=(1.04, 1), ncol=1)
#     times = np.cumsum([0] + [2*x.mean for x in self.segments])
#     for i, w in enumerate(self.sentence.words):
#         plt.text(times[i], 0.8, w.word)
#     plt.ylim(-0.05, 1.05)
#     plt.tight_layout()
#     plt.show()


if __name__ == '__main__':
    # The list is [harmony, order]
    grammar = {'det-cat': {'the': [1.0, -1]},
               'det-dog': {'the': [1.0, -1]},
               'nsubj-sleeps': {'cat': [1.0, -1], 'dog': [1.0, -1]},
               'adv-sleeps': {'quietly': [1.0, -1]}}
    wds = ['the', 'cat', 'sleeps']
    # incompat would be a list of pairs of links that are not allowed
    # to occur together
    incompat = None
    print('Processing the sentence, "{}"'.format(' '.join(wds)))
    segs = run_sent(wds, grammar, incompat, tau=1.0, T=0.5,
                    method='glauber', plot_trans=False, check_fptd=False,
                    exit_only=True, log_axis=False)
    print('Predicted mean reading times (and variances) for a range of parameters')
    ep = explore_params(wds, grammar, incompat,
                        [0.5, 1.0], [0.1, 1.0], method='glauber')
    print(ep)
