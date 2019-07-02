"""

    """


def _get_random_frame_stimuli_trials(self, frame_size, data):
    random_sequence, (movie_index, trial_index) = self._get_random_trial(data)
    batch_start = np.random.choice(range(100, random_sequence.shape[-1] - frame_size))
    frame = random_sequence[:, batch_start:batch_start + frame_size]
    image_causing_frame = self._get_y_value_for_sequence(movie_index, batch_start)

    return frame, image_causing_frame