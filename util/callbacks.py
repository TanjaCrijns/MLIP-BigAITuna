import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import Callback

class ShowSegmentation(Callback):
    """
    Keras callback to plot segmentation and labels
    for a number of samples after each epoch
    """
    def __init__(self, data_gen, n_samples=4):
        """
        # Params
        data_gen : generator which yields (img, [label, mask]) i.e. 
                   (
                       (batch_size, 3, None, None), 
                            [(batch_size, n_classes),
                             (batch_size, 1, None, None)]
                   )
                   batches. The first image of each batch will be 
                   plotted
        n_samples : number of images to show
        """
        self.data_gen = data_gen
        self.n_samples = n_samples

    def on_epoch_end(self, epoch, logs):
        for _ in range(self.n_samples):
            img, [label, mask] = next(self.data_gen)
            img, label, mask = img[0], label[0], mask[0]

            plt.subplot(131)
            plt.title('Input patch')
            plt.imshow(img.transpose(1, 2, 0) + 0.5)

            plt.subplot(132)
            plt.title('Target')
            plt.imshow(mask.squeeze())

            plt.subplot(133)
            plt.title('Segmentation')
            batch_img = np.expand_dims(img, 0)
            pred, segm = self.model.predict(batch_img, verbose=0)
            plt.imshow(segm[0].squeeze())
            plt.show()

            print 'Label:', label
            print 'Predicted:', pred[0]
            