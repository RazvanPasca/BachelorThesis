from datasets.datasets_utils import ModelType


class ModelArguments():
    def __init__(self,
                 model_type,
                 input_shape,
                 nr_filters,
                 nr_layers,
                 skip_conn_filters,
                 nr_output_classes,
                 lr,
                 clip_value,
                 reg_coeff,
                 z_dim,
                 loss):
        self.z_dim = z_dim
        self.model_type = model_type
        self.input_shape = input_shape
        self.nr_layers = nr_layers
        self.nr_filters = nr_filters
        self.skip_conn_filters = skip_conn_filters
        self.reg_coeff = reg_coeff
        self.nr_output_classes = nr_output_classes
        self.lr = lr
        self.clip_value = clip_value
        self.loss = loss

        if model_type == ModelType.NEXT_TIMESTEP:
            self.padding = 'causal'
        else:
            self.padding = 'same'
