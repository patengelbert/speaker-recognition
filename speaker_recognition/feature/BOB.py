from repoze.lru import lru_cache

import bob.ap


@lru_cache(maxsize=2)
def get_bob_extractor(fs, win_length_ms=32, win_shift_ms=16,
                      n_filters=55, n_ceps=19, f_min=0., f_max=6000,
                      delta_win=2, pre_emphasis_coef=0.95, dct_norm=True,
                      mel_scale=True):
    ret = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min,
                      f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    return ret


def extract(fs, signal=None, **kwargs):
    ret = get_bob_extractor(fs, **kwargs)(signal)
    return ret
