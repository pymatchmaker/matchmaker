import numpy as np

TOLERANCES = [50, 100, 300, 500, 1000]


def transfer_positions(wp, ref_anns, frame_rate):
    """
    Transfer the positions of the reference annotations to the target annotations using the warping path.
    Parameters
    ----------
    wp : np.array with shape (2, T)
        array of warping path.
    ref_ann : List[float]
        reference annotations in frame indices.
    """
    x, y = wp[0], wp[1]
    ref_anns_frame = np.round(ref_anns * frame_rate)
    predicted_targets = np.array([y[np.where(x >= r)[0][0]] for r in ref_anns_frame])
    return predicted_targets / frame_rate
