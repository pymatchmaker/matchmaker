import numpy as np
from hmmlearn import hmm


class DIHMM(hmm.GaussianHMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duration_dist = duration_dist
        self.interval_dist = interval_dist
        self.duration_params = np.ones(
            n_components
        )  # Default to mean duration 1 for all states
        self.interval_params = {
            "mu": np.ones(n_components),
            "sigma": np.ones(n_components),
        }  # Default parameters

    def _generate_sample_from_state(self, state, random_state=None):
        if random_state is None:
            random_state = self.random_state
        duration = self._sample_duration(state, random_state)
        interval = self._sample_interval(state, random_state)
        return (
            super()._generate_sample_from_state(state, random_state),
            duration,
            interval,
        )

    def _sample_duration(self, state, random_state):
        if self.duration_dist == "poisson":
            return random_state.poisson(self.duration_params[state])
        elif self.duration_dist == "gaussian":
            return max(
                1,
                int(
                    random_state.normal(
                        self.duration_params[state], self.duration_params[state] ** 0.5
                    )
                ),
            )
        else:
            raise ValueError(f"Unsupported duration distribution: {self.duration_dist}")

    def _sample_interval(self, state, random_state):
        if self.interval_dist == "gaussian":
            mu = self.interval_params["mu"][state]
            sigma = self.interval_params["sigma"][state]
            return max(0, int(random_state.normal(mu, sigma)))
        else:
            raise ValueError(f"Unsupported interval distribution: {self.interval_dist}")

    def fit(self, X, lengths=None):
        super().fit(X, lengths)
        self._fit_duration_distribution(X, lengths)
        self._fit_interval_distribution(X, lengths)
        return self

    def _fit_duration_distribution(self, X, lengths):
        # Implement fitting for duration distribution
        self.duration_params = np.full(
            self.n_components, np.mean(lengths) / self.n_components
        )

    def _fit_interval_distribution(self, X, lengths):
        # Implement fitting for interval distribution
        self.interval_params["mu"] = np.full(
            self.n_components, np.mean(lengths) / (2 * self.n_components)
        )
        self.interval_params["sigma"] = np.full(
            self.n_components, np.std(lengths) / (2 * self.n_components)
        )

    def score_samples(self, X):
        logprob, posteriors = super().score_samples(X)
        logprob += self._duration_logprob(X)
        logprob += self._interval_logprob(X)
        return logprob, posteriors

    def _duration_logprob(self, X):
        # Dummy implementation for duration log probabilities
        return np.zeros(X.shape[0])

    def _interval_logprob(self, X):
        # Dummy implementation for interval log probabilities
        return np.zeros(X.shape[0])


# Example usage
model = DIHMM(n_components=3, covariance_type="diag", n_iter=100)
X = np.concatenate([np.random.normal(size=(100, 2)) for _ in range(3)])
lengths = [100] * 3
model.fit(X, lengths)
