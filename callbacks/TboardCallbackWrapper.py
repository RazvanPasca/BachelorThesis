import numpy as np
from keras.callbacks import TensorBoard


class TboardCallbackWrapper(TensorBoard):
    """Sets the self.validation_data property for use with TensorBoard callback."""

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.
        self.nb_steps = nb_steps  # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs=None):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        x, y = next(self.batch_gen)
        self.validation_data = (x, y, np.ones(self.batch_size))
        return super().on_epoch_end(epoch, logs)
