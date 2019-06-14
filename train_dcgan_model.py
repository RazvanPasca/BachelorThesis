import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, CSVLogger

from WavenetImageAE.ModelTrainingParameters import ModelTrainingParameters
from WavenetImageAE.ReconstructImageCallback import ReconstructImageCallback
from WavenetImageAE.model_parameters import model_parameters
from WavenetImageAE.wavenet_dcgan_model import get_wavenet_dcgan_model
from callbacks.MetricsPlotCallback import MetricsPlotCallback
from callbacks.TboardCallbackWrapper import TboardCallbackWrapper
from utils.tf_utils import configure_gpu


def get_model_callbacks(model_params, train_images_to_reconstr, val_generator, val_images_to_reconstr):
    callbacks = []
    tensorboard_callback = TboardCallbackWrapper(batch_gen=val_generator,
                                                 nb_steps=10,
                                                 log_dir=model_params.model_path,
                                                 write_graph=True,
                                                 histogram_freq=5,
                                                 batch_size=model_params.batch_size)
    callbacks.append(tensorboard_callback)

    log_callback = CSVLogger(model_params.model_path + "/session_log.csv")
    callbacks.append(log_callback)
    
    metrics=["loss", "acc"] if "classify" in model_params.model_output_type else ["loss"]
    metric_callback = MetricsPlotCallback(model_params.model_path, metrics)
    callbacks.append(metric_callback)

    save_model_callback = ModelCheckpoint(filepath="{}/best_model.h5".format(model_params.model_path),
                                          monitor="val_loss",
                                          save_best_only=True)
    callbacks.append(save_model_callback)

    if model_params.model_output_type.upper() == "DCGAN":
        reconstruct_image_callback = ReconstructImageCallback(train_images_to_reconstr,
                                                              val_images_to_reconstr,
                                                              model_params.logging_period,
                                                              model_params.nr_rec,
                                                              model_params.model_path)
        callbacks.append(reconstruct_image_callback)
    return callbacks


def train_model(model_params):
    model = get_wavenet_dcgan_model(nr_filters=model_params.nr_filters,
                                    input_shape=(model_params.input_shape, 47),
                                    nr_layers=model_params.nr_layers,
                                    lr=model_params.lr,
                                    loss=model_params.loss,
                                    clipvalue=model_params.clip_grad_by_value,
                                    skip_conn_filters=model_params.skip_conn_filters,
                                    regularization_coef=model_params.regularization_coef,
                                    z_dim=model_params.z_dim,
                                    output_type=model_params.model_output_type,
                                    nr_classes=model_params.dataset.nr_classes)

    print(model.summary())

    train_generator = model_params.dataset.train_frame_generator(model_params.frame_size, model_params.batch_size)
    val_generator = model_params.dataset.validation_frame_generator(model_params.frame_size, model_params.batch_size)

    train_images_to_reconstr = None
    val_images_to_reconstr = None

    if model_params.model_output_type.upper() == "DCGAN":
        train_images_to_reconstr = next(train_generator)
        val_images_to_reconstr = next(val_generator)

    callbacks = get_model_callbacks(model_params, train_images_to_reconstr, val_generator, val_images_to_reconstr)
    print("Steps per train {} |  Steps per val {} ".format(model_params.nr_train_steps, model_params.nr_val_steps))

    plt.hist([movie for movie in model_params.dataset.stimuli_mean], bins=30,
             label=[str(i) for i in model_params.movies_to_keep])
    plt.legend(prop={'size': 10})
    plt.savefig("{}/movies_brightness_histo.png".format(model_params.model_path), format="png")

    model.fit_generator(train_generator,
                        steps_per_epoch=model_params.nr_train_steps,
                        epochs=model_params.n_epochs,
                        validation_data=val_generator,
                        validation_steps=model_params.nr_val_steps,
                        verbose=2,
                        use_multiprocessing=True,
                        callbacks=callbacks)

    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


if __name__ == '__main__':
    model_parameters = ModelTrainingParameters(model_parameters)
    configure_gpu(model_parameters.gpu)
    train_model(model_parameters)
    # test_model(model_parameters)
