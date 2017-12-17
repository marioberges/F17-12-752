"""Extended viterbi algorithm."""

import numpy as np
from pandas import get_dummies
from scipy import stats
from typing import Iterable, Callable, Sized, Any, Dict
from itertools import zip_longest, product, tee
from functools import partial, reduce
from .helpers import dict_zip

StatePars = Dict[int, Dict[str, float]]


def viterbi(obs: Iterable, features,
            fn_tr: Callable, initps: Sized,
            emx: Callable, emy: Callable, emyp_threshold: float =0.01) -> Iterable:
    states = get_dummies(range(len(initps)))

    normalize = lambda vec: vec / np.sum(vec)

    def forward():
        start = normalize(np.array(initps) * emx(obs[0][0]))

        yield [*zip_longest(start, [None])]
        last_ps = start
        for ob, feature in zip(obs[1:], features[:-1]):
            xmat = np.hstack([np.tile(feature, (len(initps), 1)),
                              states])
            _yemp = emy(ob[1])
            yemp = _yemp if np.any(_yemp > emyp_threshold) else np.ones_like(_yemp)
            all_paths = last_ps * fn_tr(xmat).T * yemp
            this_ps = normalize(all_paths.max(axis=1) * emx(ob[0]))  # normalize to avoiding rounding errors
            yield [*zip(this_ps, all_paths.argmax(axis=1))]

            last_ps = this_ps

    def backtrack():
        viterbi_mat = reversed([*forward()])
        last, (_, prev) = max(enumerate(next(viterbi_mat)),
                              key=lambda x: x[1][0])
        yield last
        for step in viterbi_mat:
            yield prev
            _, prev = step[prev]

    return reversed([*backtrack()])


def _get_emit_fns(state_pars: StatePars) -> Callable[[Any], float]:
    xfns = {k: partial(stats.norm.cdf, **v)
            for k, v
            in state_pars.items()}

    def gen_yfn(pars):
        that, this = pars
        arg_mergers = {'loc': lambda x, y: y - x,
                       'scale': lambda x, y: np.sqrt(x ** 2 + y ** 2)}
        merged_args = {k: v[1](*v[0])
                       for k, v
                       in reduce(dict_zip, [that[1],
                                            this[1],
                                            arg_mergers]).items()}
        yfn = partial(stats.norm.pdf, **merged_args)
        return (that[0], this[0]), yfn

    yfns = dict(map(gen_yfn, product(*tee(state_pars.items(), 2))))
    return xfns, yfns


def get_npem_fns(state_pars: StatePars) -> Callable:
    all_states = [*range(len(state_pars))]
    xfns, yfns = _get_emit_fns(state_pars)
    emx = lambda ob: np.array([*map(lambda s: xfns[s](ob),
                                    all_states)])
    emy = lambda ob: np.array([*map(lambda s1: [*map(lambda s2: yfns[(s1, s2)](ob),
                                                     all_states)],
                                    all_states)])

    return emx, emy
