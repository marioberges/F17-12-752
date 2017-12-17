"""Calculate transition probabilities and initial probabilities."""

from collections import Counter
from sklearn.linear_model import LogisticRegression
from typing import Callable, Tuple
from pandas import get_dummies
import numpy as np


def softmax(xmat,
            this_y, next_y,
            *args, **kwargs) -> Tuple[Callable, Callable]:
    model_init = LogisticRegression(*args, **kwargs)
    model_init.fit(xmat, this_y)

    model_trans = LogisticRegression(*args, **kwargs)
    model_trans.fit(np.hstack([xmat, get_dummies(this_y)]),
                    next_y)

    def init_proba(xvec):
        _xvec = xvec.reshape(1, -1) if len(xvec.shape) == 1 else xvec
        return model_init.predict_proba(_xvec).reshape(-1, )

    return init_proba, model_trans.predict_proba
