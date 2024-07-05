# %%
import numpy as np
from hmmlearn import hmm
from scipy.sparse import lil_matrix
from scipy.stats import norm

# %%
# Chroma 데이터를 위한 가상 예시
# 실제 데이터는 피아노 롤 또는 chroma 벡터 등의 형식이어야 합니다.
score_chroma = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C#
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # D
        # ... 기타 음표
    ]
)
performance_chroma = np.array(
    [
        [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 거의 C
        [0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 거의 C#
        [0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],  # 거의 D
        # ... 기타 연주 데이터
    ]
)

# HMM 모델 정의
n_components = len(score_chroma)  # 상태의 개수는 악보의 음표 개수와 동일
model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)

# 모델 학습 (이 과정에서 실제 데이터는 더 정교하게 필요)
model.fit(score_chroma)

# 예측
logprob, state_sequence = model.decode(performance_chroma)

print("정렬된 상태 시퀀스:", state_sequence)

# %%
import numpy as np
from hmmlearn import hmm

# Example chroma feature data for score (with sustained notes)
score_chroma = np.array(
    [
        [0.8, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C
        [0.8, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C sustained
        [0.8, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C sustained
        [0.8, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C sustained
        [0, 0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C#
        [0, 0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # C# sustained
        [0.1, 0, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],  # D
        [0.1, 0, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],  # D sustained
        [0.1, 0, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],  # D sustained
        [0.1, 0, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],  # D sustained
        [0, 0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0],  # D#
        [0, 0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0],  # D# sustained
        [0, 0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0],  # D# sustained
        [0, 0.1, 0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0],  # D# sustained
        [0.1, 0, 0, 0, 0.8, 0.1, 0, 0, 0, 0, 0, 0],  # E
    ]
)

# Example chroma feature data for performance (with slight variations and different length)
performance_chroma = np.array(
    [
        [0.75, 0.15, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Almost C
        [0.75, 0.15, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Almost C sustained
        [0.8, 0.15, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Slightly different C sustained
        [0, 0.85, 0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Almost C#
        [0.05, 0.85, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Slightly different C# sustained
        [0.05, 0, 0.9, 0.05, 0, 0, 0, 0, 0, 0, 0, 0],  # Almost D
        [0.05, 0, 0.9, 0.05, 0, 0, 0, 0, 0, 0, 0, 0],  # Almost D sustained
        [0.1, 0, 0.85, 0.05, 0, 0, 0, 0, 0, 0, 0, 0],  # Slightly different D sustained
        [0.1, 0, 0.85, 0.05, 0, 0, 0, 0, 0, 0, 0, 0],  # Slightly different D sustained
        [0, 0.1, 0, 0.85, 0.05, 0, 0, 0, 0, 0, 0, 0],  # Almost D#
        [0, 0.1, 0, 0.85, 0.05, 0, 0, 0, 0, 0, 0, 0],  # Almost D# sustained
        [0, 0.1, 0, 0.85, 0.05, 0, 0, 0, 0, 0, 0, 0],  # Almost D# sustained
        [0.05, 0.15, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0],  # Slightly different D# sustained
        [0.1, 0, 0, 0, 0.75, 0.15, 0, 0, 0, 0, 0, 0],  # Almost E
    ]
)

# Define the number of states (note, age) pairs
num_notes = 5
max_age = 4  # Example max age for each note
n_components = num_notes * max_age

# Define transition matrix with higher probability of transitioning to the next state as age increases
transmat = np.zeros((n_components, n_components))
for i in range(num_notes):
    for j in range(max_age):
        # 동일한 노트에서 나이가 증가할 때의 전이 확률
        if j < max_age - 1:
            transmat[i * max_age + j, i * max_age + j + 1] = 0.9 - 0.1 * j
        # 다른 노트로 전환될 때의 전이 확률
        if i < num_notes - 1:
            transmat[i * max_age + j, (i + 1) * max_age] = 0.1 + 0.1 * j

# Adding a small probability to avoid zero rows
epsilon = 1e-6
transmat += epsilon

# Renormalizing the transition matrix
transmat /= transmat.sum(axis=1, keepdims=True)

# Initial probabilities, assuming the performance starts at the first note
startprob = np.zeros(n_components)
startprob[:max_age] = 1.0 / max_age

# Define HMM model with Gaussian emissions
model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
model.startprob_ = startprob
model.transmat_ = transmat

# Train the model
model.fit(np.tile(score_chroma, (max_age, 1)))

# Decode the performance data
logprob, state_sequence = model.decode(performance_chroma)

# Convert state sequence to (note, age) pairs
note_age_sequence = [
    (state // max_age, state % max_age + 1) for state in state_sequence
]

print("Note-Age sequence:", note_age_sequence)


# %%
def jiang_transition_matrix_audio(
    n_notes: int,
    total_frames: int,
    frame_rate: float,
    sigma: float,
    transition_variance: float,
) -> lil_matrix:
    """Compute the transition matrix with merging nodes for the Jiang HMM model.

    Args:
        n_notes (int): Number of note indices in the HMM
        total_frames (int): Total number of frames
        frame_rate (float): Frame rate in Hz
        sigma (float): Standard deviation for note duration
        transition_variance (float): Variance for the transition matrix

    Returns:
        lil_matrix: A sparse matrix representing the transition probabilities.
    """
    transition_matrix = lil_matrix((total_frames, total_frames))
    delta = 1 / frame_rate

    for frame in range(total_frames - 1):
        note_index = frame // (total_frames // n_notes)
        note_age = frame % (total_frames // n_notes)

        # Calculate the mean duration based on the note index
        mean_duration = (note_index + 1) * delta
        stddev_duration = sigma

        if stddev_duration == 0:
            stddev_duration = 1e-6  # Prevent division by zero

        # Calculate the Gaussian CDF terms for stay probability
        phi_current = norm.cdf((note_age * delta - mean_duration) / stddev_duration)
        phi_next = norm.cdf(((note_age + 1) * delta - mean_duration) / stddev_duration)

        if (1 - phi_current) == 0:
            p1 = 1  # Prevent division by zero
        else:
            p1 = (1 - phi_next) / (1 - phi_current)

        # Ensure p1 is between 0 and 1
        p1 = max(0, min(p1, 1))

        # Transition within the same frame (staying in the same state)
        transition_matrix[frame, frame] = p1

        # Transition to the next frame (moving to the next state)
        if frame + 1 < total_frames:
            transition_matrix[frame, frame + 1] = 1 - p1

        # Merging similar states by combining transitions with similar probabilities
        if note_age == 0 and frame + (total_frames // n_notes) < total_frames:
            for k in range(1, total_frames // n_notes):
                next_frame = frame + k
                next_note_index = next_frame // (total_frames // n_notes)
                if next_note_index == note_index + 1:
                    transition_matrix[frame, next_frame] += (1 - p1) / (
                        total_frames // n_notes - 1
                    )

    # Ensure the last state transitions to itself
    transition_matrix[total_frames - 1, total_frames - 1] = 1.0

    return transition_matrix


# %%
# Example usage
n_notes = 10  # Example number of notes
max_duration = 20  # Example maximum duration each note can last (in frames)
frame_rate = 30  # Example frame rate in Hz
sigma = 0.1  # Example standard deviation for note duration
transition_variance = 1.0  # Example variance for the transition matrix

transition_matrix = jiang_transition_matrix_audio(
    n_notes, max_duration, frame_rate, sigma, transition_variance
)

print(transition_matrix.toarray())

# %%
