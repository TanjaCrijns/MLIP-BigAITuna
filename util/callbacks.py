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
        i = 0
        for img_batch, y_batch in self.data_gen:
            for img, y in zip(img_batch, y_batch):
                if type(y) is tuple:
                    label, mask = y
                else:
                    label = None
                    mask = y
                plt.figure(figsize=(13, 7))
                plt.subplot(131)
                plt.title('Input patch')
                plt.imshow(img.transpose(1, 2, 0) + 0.5)

                plt.subplot(132)
                plt.title('Target')
                plt.imshow(mask[1])

                plt.subplot(133)
                plt.title('Segmentation')
                batch_img = np.expand_dims(img, 0)
                pred = self.model.predict(batch_img, verbose=0)
                if type(pred) is tuple:
                    pred, segm = pred
                else:
                    segm = pred
                plt.imshow(segm[0, 1])
                plt.show()

                if label is not None:
                    print 'Label:', label
                    print 'Predicted:', pred[0]
                i += 1
                if i >= self.n_samples:
                    return

            