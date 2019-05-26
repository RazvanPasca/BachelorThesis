from keras.callbacks import ModelCheckpoint, CSVLogger

from WavenetImageAE.ModelTrainingParameters import ModelTrainingParameters
from WavenetImageAE.ReconstructImageCallback import ReconstructImageCallback
from WavenetImageAE.model_parameters import model_parameters
from WavenetImageAE.wavenet_dcgan_model import get_wavenet_dcgan_model
from callbacks.MetricsPlotCallback import MetricsPlotCallback
from callbacks.TboardCallbackWrapper import TboardCallbackWrapper
from utils.tf_utils import configure_gpu


def train_model(model_params):
    model = get_wavenet_dcgan_model(nr_filters=model_params.nr_filters,
                                    input_shape=(model_params.frame_size, 47),
                                    nr_layers=model_params.nr_layers,
                                    lr=model_params.lr,
                                    loss=model_params.loss,
                                    clipvalue=model_params.clip_grad_by_value,
                                    skip_conn_filters=model_params.skip_conn_filters,
                                    regularization_coef=model_params.regularization_coef,
                                    z_dim=model_params.z_dim)

    print(model.summary())

    val_generator = model_params.dataset.validation_frame_generator(model_params.frame_size, model_params.batch_size)
    train_generator = model_params.dataset.train_frame_generator(model_params.frame_size, model_params.batch_size)

    train_images_to_reconstr = next(train_generator)
    val_images_to_reconstr = next(val_generator)

    tensorboard_callback = TboardCallbackWrapper(batch_gen=val_generator,
                                                 nb_steps=10,
                                                 log_dir=model_params.model_path,
                                                 write_graph=True,
                                                 histogram_freq=5,
                                                 batch_size=model_params.batch_size)

    log_callback = CSVLogger(model_params.model_path + "/session_log.csv")

    reconstruct_image_callback = ReconstructImageCallback(train_images_to_reconstr,
                                                          val_images_to_reconstr,
                                                          model_params.logging_period,
                                                          model_params.nr_rec,
                                                          model_params.model_path)
    metric_callback = MetricsPlotCallback(model_params.model_path, graphs=["loss"])

    save_model_callback = ModelCheckpoint(filepath="{}/best_model.h5".format(model_params.model_path),
                                          monitor="val_loss",
                                          save_best_only=True)

    model.fit_generator(train_generator,
                        steps_per_epoch=model_params.nr_train_steps,
                        epochs=model_params.n_epochs,
                        validation_data=val_generator,
                        validation_steps=model_params.nr_val_steps,
                        verbose=2,
                        use_multiprocessing=True,
                        callbacks=[tensorboard_callback, save_model_callback, metric_callback, log_callback,
                                   reconstruct_image_callback])

    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


if __name__ == '__main__':
    model_parameters = ModelTrainingParameters(model_parameters)
    configure_gpu(model_parameters.gpu)
    train_model(model_parameters)
    # test_model(model_parameters)
