import os
from glob import glob
import cv2
import numpy as np
from shutil import rmtree
from keras.callbacks import Callback
from time import time

from utils import preprocess_LR, deprocess_HR


class BSD100_Evaluator(Callback):
    def __init__(self, directory, filepath=None, downscale_factor=4, color_mode='rgb',
                 verbose=1, data_format='channels_last', margin=5):
        """
        Keras Callback to use for vaidation on BSD100 dataset.

        :param directory: path to the BSD100 dataset directory
        :param filepath: directory where LR|SR|HR images will be saved. If None, it does not save images.
        :param downscale_factor: downscale factor to apply to LR images
        :param color_mode: RGB or BGR mode
        :param verbose: verbose level. 0 or 1
        :param data_format: order of dimensions for TensorFlow
        :param margin: remove margin pixels strips from each border
        """
        self.directory = directory
        self.filepath = filepath
        self.downscale_factor = downscale_factor
        self.color_mode = color_mode
        self.verbose = verbose
        self.data_format = data_format
        self.margin = margin
        self.slice = np.s_[margin:-margin, margin:-margin,:]

        # list files
        self.list_files_HR = glob(os.path.join(directory, "*_HR.png"))
        print("Found {} images ...".format(len(self.list_files_HR)))

        # test if filepath exists, if true remove it
        # then create an empty dir
        if self.filepath is not None:
            if os.path.isdir(self.filepath):
                rmtree(self.filepath)
            os.mkdir(self.filepath)
        
        self.logs = dict()
        self.logs['mse'] = []
        self.logs['psnr'] = []

        super(BSD100_Evaluator, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        list_mse = []
        list_psnr = []

        current_filepath = os.path.join(self.filepath, 'epoch_{}'.format(epoch+1))
        if os.path.isdir(current_filepath):
            rmtree(current_filepath)
        os.mkdir(current_filepath)

        start = time()
        for i, file_HR in enumerate(self.list_files_HR):
            img_hr = cv2.imread(file_HR, cv2.IMREAD_COLOR)
            img_name = file_HR.split('/')[-1].split('_')[0]

            if self.color_mode == 'bgr':
                img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

            new_shape = img_hr.shape

            img_lr = cv2.resize(img_hr, (new_shape[1] // self.downscale_factor,
                                         new_shape[0] // self.downscale_factor), interpolation=cv2.INTER_LINEAR)

            img_cubic = cv2.resize(img_lr, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC)

            img_lr = preprocess_LR(img_lr)

            if self.data_format == 'channels_first':
                # (h,w,c) to (c,h,w)
                img_lr = np.transpose(img_lr, (2, 0, 1))

            img_sr = self.model.predict_on_batch(np.expand_dims(img_lr, axis=0))[0]

            if self.data_format == 'channels_first':
                # (c,h,w) to (h,w,c)
                img_sr = np.transpose(img_sr, (1, 2, 0))

            # compute MSE (with images in range [0, 255])
            img_sr = deprocess_HR(img_sr)
            mse = np.mean(np.square(img_hr[self.slice] - img_sr[self.slice]))
            list_mse.append(mse)

            # compute PSNR
            if mse == 0.:
                psnr = 100
            else:
                psnr = 20 * np.log10(255. / np.sqrt(mse))
            list_psnr.append(psnr)

            if self.filepath is not None:
                global_image = np.zeros((new_shape[0], 3*new_shape[1], 3), dtype=np.uint8)
                global_image[:, 0:new_shape[1], :] = img_cubic.astype(np.uint8)
                global_image[:, new_shape[1]:2*new_shape[1], :] = img_sr.astype(np.uint8)
                global_image[:, 2*new_shape[1]:3*new_shape[1], :] = img_hr.astype(np.uint8)

                # add black padding
                global_image_ext = np.zeros((global_image.shape[0]+50, global_image.shape[1], 3), dtype=np.uint8)
                global_image_ext[0:global_image.shape[0], :, :] = global_image
                global_image_ext = cv2.putText(img=np.copy(global_image_ext),
                                               text="MSE = {:.3f} | PSNR = {:.3f}".format(mse, psnr),
                                               org=(0, new_shape[0]+50),
                                               fontFace=1,
                                               fontScale=2,
                                               color=(255, 255, 255),
                                               thickness=2)

                cv2.imwrite(os.path.join(current_filepath, '{}.png'.format(img_name)), global_image_ext.astype(np.uint8),
                            params=[cv2.IMWRITE_PNG_COMPRESSION, 3])
        
        self.logs['mse'].append(np.mean(list_mse))
        self.logs['psnr'].append(np.mean(list_psnr))
        stop = time()
        
        if self.verbose > 0:
            print('\nBSD100 Callback - Epoch %05d: MSE = %s  || PSNR = %s  in %05d s' % (epoch + 1, np.mean(list_mse),
                                                                                         np.mean(list_psnr),
                                                                                         stop-start))


def BSD100_evaluate(model, step, directory, filepath=None, verbose=1, color_mode='rgb',
                    downscale_factor=4, margin=5, data_format='channels_last'):
    """
    :param model: model to evaluate on BSD100
    :param step: step id. Used if filepath is not None.
    :param directory: path to the BSD100 dataset directory
    :param filepath: directory where LR|SR|HR images will be saved. If None, it does not save images.
    :param downscale_factor: downscale factor to apply to LR images
    :param color_mode: RGB or BGR mode
    :param verbose: verbose level. 0 or 1
    :param data_format: order of dimensions for TensorFlow
    :param margin: remove margin pixels strips from each border
    """

    list_files_HR = glob(os.path.join(directory, "*_HR.png"))
    print("Found {} images ...".format(len(list_files_HR)))

    list_mse = []
    list_psnr = []

    slice = np.s_[margin:-margin, margin:-margin, :]

    current_filepath = os.path.join(filepath, 'epoch_{}'.format(step))
    if filepath is not None:
        if os.path.isdir(current_filepath):
            rmtree(current_filepath)
        os.mkdir(current_filepath)

    start = time()
    for i, file_HR in enumerate(list_files_HR):
        img_hr = cv2.imread(file_HR, cv2.IMREAD_COLOR)
        img_name = file_HR.split('/')[-1].split('_')[0]

        if color_mode == 'bgr':
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)

        new_shape = img_hr.shape

        img_lr = cv2.resize(img_hr, (new_shape[1] // downscale_factor,
                                     new_shape[0] // downscale_factor), interpolation=cv2.INTER_LINEAR)

        img_cubic = cv2.resize(img_lr, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC)

        img_lr = preprocess_LR(img_lr)

        if data_format == 'channels_first':
            # (h,w,c) to (c,h,w)
            img_lr = np.transpose(img_lr, (2, 0, 1))

        img_sr = model.predict_on_batch(np.expand_dims(img_lr, axis=0))[0]

        if data_format == 'channels_first':
            # (c,h,w) to (h,w,c)
            img_sr = np.transpose(img_sr, (1, 2, 0))

        # compute MSE (with images in range [0, 255])
        img_sr = deprocess_HR(img_sr)
        mse = np.mean(np.square(img_hr[slice] - img_sr[slice]))
        list_mse.append(mse)

        # compute PSNR
        if mse == 0.:
            psnr = 100
        else:
            psnr = 20 * np.log10(255. / np.sqrt(mse))
        list_psnr.append(psnr)

        if filepath is not None:
            global_image = np.zeros((new_shape[0], 3 * new_shape[1], 3), dtype=np.uint8)
            global_image[:, 0:new_shape[1], :] = img_cubic.astype(np.uint8)
            global_image[:, new_shape[1]:2 * new_shape[1], :] = img_sr.astype(np.uint8)
            global_image[:, 2 * new_shape[1]:3 * new_shape[1], :] = img_hr.astype(np.uint8)

            # add black padding
            global_image_ext = np.zeros((global_image.shape[0] + 50, global_image.shape[1], 3), dtype=np.uint8)
            global_image_ext[0:global_image.shape[0], :, :] = global_image
            global_image_ext = cv2.putText(img=np.copy(global_image_ext),
                                           text="MSE = {:.3f} | PSNR = {:.3f}".format(mse, psnr),
                                           org=(0, new_shape[0] + 50),
                                           fontFace=1,
                                           fontScale=2,
                                           color=(255, 255, 255),
                                           thickness=2)

            cv2.imwrite(os.path.join(current_filepath, '{}.png'.format(img_name)), global_image_ext.astype(np.uint8),
                        params=[cv2.IMWRITE_PNG_COMPRESSION, 3])
    stop = time()

    if verbose > 0:
        print('\nBSD100 Callback - Epoch %05d: MSE = %s  || PSNR = %s  in %05d s' % (step, np.mean(list_mse),
                                                                                     np.mean(list_psnr),
                                                                                     stop - start))
    return np.mean(list_mse), np.mean(list_psnr)
