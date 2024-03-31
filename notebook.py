# %%
from matchmaker.dp.oltw_jiyun import (
    OLTW,
    StreamProcessor,
    SAMPLE_RATE,
    HOP_LENGTH,
)
import librosa
import matplotlib.pyplot as plt
import numpy as np
import traceback
from IPython.display import display
import scipy
from pathlib import Path
import pandas as pd
from infer import convert_score_to_audio, extract_beats_and_downbeats

# %%
# Run OLTW with mock audio
ref_audio_path = (
    "/Users/jiyun/workspace/asap-dataset/Bach/Fugue/bwv_862/midi_score_adjusted.wav"
)
target_audio_path = "/Users/jiyun/workspace/asap-dataset/Bach/Fugue/bwv_862/Song04M.wav"
duration = int(librosa.get_duration(path=ref_audio_path)) + 2

oltw = OLTW(StreamProcessor(), ref_audio_path)
fig = plt.figure()
ax = fig.gca()
hfig = display(fig, display_id=True)
h = ax.imshow(
    np.zeros((12, int(SAMPLE_RATE / HOP_LENGTH) * duration)),
    aspect="auto",
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

try:
    oltw.run(fig, h, hfig, mock=True, mock_audio_path=target_audio_path)
except Exception as e:
    print(f"error! : {str(e)}, {type(e)}")
    traceback.print_tb(e.__traceback__)
    oltw.stop()

print(f"=====================oltw run ended=====================")

# %%
dist = scipy.spatial.distance.cdist(
    oltw.ref_features.T, oltw.target_features[:, : oltw.target_pointer].T
)  # [d, wy]
plt.figure(figsize=(20, 20))
plt.imshow(dist.T, aspect="auto", origin="lower", interpolation="nearest")
x, y = zip(*oltw.candi_history)

from matplotlib import cm

cmap = cm.get_cmap("magma", 100)
for n in range(len(x)):
    plt.plot(x[n], y[n], ".", color="r")

# %%
# RUN ASAP EVALUATION
asap_path = Path("/Users/jiyun/workspace/asap-dataset")
test_metadata = pd.read_csv(
    "/Users/jiyun/workspace/ismir2024_matchmaker/data/metadata-asap-test.csv"
)
for row in test_metadata.itertuples():
    target_audio_path = asap_path / row.audio_performance
    score = asap_path / row.xml_score
    print(row)

# %%
