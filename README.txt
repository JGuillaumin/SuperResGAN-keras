
# Single Image Super-Resolution using GANs - Keras implementation 

This project implements, with Keras (Deep Learning framework), the approaches developed in _Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network_ from Twitter ([paper here](https://arxiv.org/abs/1609.04802). 


### Authors 

This work was done during a research project at IMT Atlantique. 

- Fatimazahra IMANI (fatimazahra.imani `at` imt-atlantique `dot` net)
- Julien GUILLAUMIN (julien.guillaumin `at` imt-atlantique `dot` net)


### Paper summary:

In this paper, the **Generative Adversarial Networks framework** is train a neural network to perform **super-resolution on optical images**.
The goal is to convert a low resolution image, 64x64 pixels, to an high resolution version. 
This conversion is done by a first neural network, the generator *G* which produces super-resolution (_SR_) version of low-resolution (_LR_) images. 

This network can be trained by minimizing a Mean Squared Error (MSE) between the SR images and the high-resolution images (HR).
Unfortunately it outputs blurred images with visual artifacts. 

So they add a second loss: **Perceptual loss function**.  
This loss uses a pre-trained (and freezed) VGG network (pretraining performed on ImageNet dataset).
This additional objective is inspired from previous work on neural style transfer (see [this paper][1] as initial paper in this topic)

The goal is to obtain SR images with similar feature maps (extracted from the VGG) as the feature maps of the HR images.
So it adds a new MSE loss between elements of a _learned feature space_ (while the first MSE loss is between objects defined in the _image space_)
This feature space corresponds to the output of given convolutional layer of the VGG. 
Why ? It helps the generator G to produce SR images that follows the high-level representations of HR images (high-level means features from deep layers in VGG network).

Even with this additional loss, produced images are blurred, and it's **easy to distinguish SR images**. 
With our perceptual loss, it's easy to make this distinction: SR vs HR.
So we need a loss, that penalize G when it produces non realistic SR images: here we are -> the Generative Adversarial Networks framework !

We will add a new network, the **discriminator D**, which takes as input SR and HR images (same dimension) and it predicts the classes **real vs fake**.
In our case: **real=HR** and **fake=SR**. The discriminator and the generator are trained together : one step we will train D to make the distinction between SH and HR images,
the next step we will **train G to mislead D**. 

This new loss helps G to produce images in **the manifold of realistic images** !
In practice, D finds out this manifold and says to G if it produces images in this subspace of the image space.
At the same time D is trained to get a better estimation of this manifold.



All this approaches requires a training dataset of LR/HR images. As in the paper, we used the [COCO dataset][2].
It contains about 80k images (We used only the train part of the dataset.)
To evaluate the performances, as in the paper, we used the [BSD100][3] dataset.
The metric computations are integrated in a Keras callback.

### Implemented features 

- Efficient DataGenerator for COCO dataset
- Automatic `data_format` selection ! (**from 1O to 30% faster**)
- A custom Callback that computes specific metrics on BSD100 images 
- How to use a pre-trained VGG within a loss 
- How to compute the perceptual loss from neural style transfer
- How to combine several losses
- How to train GANs with Keras

**Note**: `ShufflePixel` is replaced by `UpSampling2D` layers !

### Source Code

**About the implementation**:
The network definitions are duplicated in the several notebooks. The reason is to have a step-by-step notebook with all relevant code ine one file. Useful for educational purposes.
(and we are working with Keras, so no need a lot of code!)

- `batch_generator.py`: efficient batch generator for COCO dataset.
- `bsd100_callbacks.py` : Keras callbacks that computes the PSNR and saves SR images from [BSD100][3] dataset.
- `ùtils.py` :  methods that perform preprocessing and deprocessing on images (due to the VGG)

- `SRResNet-MSE.ipynb`: define and train G only with a MSE loss
- `SRResNet-VGG22.ipynb`: define and train G with MSE and perceptual loss (features from `block2_conv2`)
- `SRResNet-VGG54.ipynb`: define and train G with MSE and perceptual loss (features from `block5_conv4`)

- `SRGAN-MSE.ipynb` : define and train G + D. G is trained also with a MSE loss.
- `SRGAN-VGG22.ipynb` : define and train G + D. G is trained also with a perceptual loss (features from `block2_conv2`).
- `SRGAN-VGG54.ipynb` : define and train G + D. G is trained also with a perceptual loss (features from `block5_conv4`).


### About the `COCOBatchGenerator` 

See `batch_generator.py` for code.

This batch generator is inspired from the classical batch generators available in Keras. 
Here, we created a special batch generator that outputs batches of SR images (inputs) and their HR versions (targets). 

```python
from batch_generator import COCOBatchGenerator

batch_gen = COCOBatchGenerator('data/COCO/train2014/',
                               target_size=(256,256),
                               downscale_factor=4, 
                               batch_size=8,
                               shuffle=True,
                               seed=None,
                               color_mode='rgb',
                               crop_mode='fixed_size',
                               data_format='channels_last')
                               
batch_LR, batch_HR = batch_gen.next()
```

**`crop_mode`**: `fixed_size` vs `random_size`:

- `fixed_size` : with this mode, the batch generator will randomly crop a patch of size `(256,256)` in the image. If the image is too small, it will crop 
a patch with size `(short_edge, short_edge)` and will be resized to `(256,256)` (`interpolation=cv2.INTER_CUBIC`).

- `random_size`: in this mode, the batch generator will randomly crop a patch with a random shape (within the range `[256, short_edge]`). The cropped patch will be resized to `(256, 256)` with OpenCV (`interpolation=cv2.INTER_CUBIC`).



### References
- [1] : https://arxiv.org/abs/1508.06576
- [2] : http://cocodataset.org/
- [3] : http://www.ifp.illinois.edu/~dingliu2/iccv15/html/SRdemoFrame_BSD100.html