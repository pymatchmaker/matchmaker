from pathlib import Path

import librosa
import mido
import numpy as np
import partitura
from partitura import save_wav_fluidsynth

from matchmaker.features.audio import SAMPLE_RATE

TOLERANCES = [50, 100, 300, 500, 1000, 2000]


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


def adjust_tempo_for_performance_audio(score_part, performance_audio: Path):
    default_tempo = 120
    score_midi = partitura.save_score_midi(score_part, out=None)
    source_length = score_midi.length
    target_length = librosa.get_duration(path=str(performance_audio))
    ratio = target_length / source_length
    rounded_tempo = int(round(default_tempo / ratio, -1))  # round to the nearest 10
    print(
        f"default tempo: {default_tempo} (score length: {source_length}) -> adjusted_tempo: {rounded_tempo} (perf length: {target_length})"
    )
    return rounded_tempo


def regenerate_tempo_adjusted_midi(midi_path: Path, target_duration: float) -> Path:
    mid = mido.MidiFile(midi_path.as_posix())
    ratio = target_duration / mid.length
    print(f"Tempo ratio (target audio / midi length): {ratio}")

    has_set_tempo = False
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                has_set_tempo = True
                print(f"Original tempo: {mido.tempo2bpm(msg.tempo)}, {msg.tempo}")
                new_tempo = mido.bpm2tempo(mido.tempo2bpm(msg.tempo) / ratio)
                print(f"New tempo: {mido.tempo2bpm(new_tempo)}, {new_tempo}")
                msg.tempo = int(new_tempo)
    if not has_set_tempo:
        # If the track does not have a 'set_tempo' message, add one with the default BPM
        default_bpm = 120
        print(f"[Default] Original tempo: {default_bpm}, 500000")
        new_tempo = mido.bpm2tempo(default_bpm / ratio)
        print(f"New tempo: {new_tempo}")
        for track in mid.tracks:
            track.insert(0, mido.MetaMessage("set_tempo", tempo=int(new_tempo)))

    new_midi_path = midi_path.with_stem("midi_score_adjusted")
    mid.save(new_midi_path)
    return new_midi_path
