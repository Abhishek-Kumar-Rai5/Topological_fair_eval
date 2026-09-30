"""Train / validation split (prereg §2).

Validation = the LAST 20% of each machine's training data, in time order.
No shuffling: a time series must not leak future points into training.
Training data is treated as normal, so the validation windows are all normal too.
"""


def split_train_val(train, val_frac=0.2):
    """Split a (time, channels) array into (fit_part, validation_part).

    fit_part:        first (1 - val_frac) of the rows, used to fit scaler + detector
    validation_part: last val_frac of the rows, used only to set the threshold
    """
    assert 0 < val_frac < 1
    n_val = int(round(len(train) * val_frac))
    cut = len(train) - n_val
    return train[:cut], train[cut:]
