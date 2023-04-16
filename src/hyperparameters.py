from dataclasses import dataclass


@dataclass
class Hparams:
    # dataloader params
    dataset_dir: str = "dataset/mvtec_anomaly_detection"
    obj_classes: int = 15 # number of classes in the dataset
    augmentation: bool = False # apply augmentation startegy to input images
    version: str = "2" # versions of CAE, 1 does use flat latent space, 2 multidimensional
    img_size: int = 224  # size of image in v1 256 works better, in v2 224
    img_channels: int = 3 # RGB channels
    batch_size: int = 44 # size of the batches
    n_cpu: int = 12  # number of cpu threads to use for the dataloaders
    pin_memory: bool = False # parameter to pin memory in dataloader
    
    # AE params
    latent_size: int = 128
    anomaly_strategy: str = "ssim" # "mssim", "mse", "ssim"
    training_strategy: str = "ssim" # "mssim", "mse", "ssim"
    lr: float = 2e-4 # 2e-4 or 1e-3
    threshold: float = 0.5 # initialization of the threshold
    gaussian_initialization: bool = True # perform or not the Gaussian inizialization
    t_weight: float = 0.65 # how much weight the new threshold wrt the old
    loss_weight: float = 1 # how much weight the reconstruction loss between two pixels 
    min_lr: float = 1e-8 # min lr for ReduceLROnPlateau
    adam_eps: float = 1e-6 # term added to the denominator to improve numerical stability
    w_std: float = +0.3 # this param weights how much we are going to add of the std in treshold update
    wd: float = 1e-6 # weight decay as regulation strategy
    noise: float = 0.4 # noise factor in (0,1) for the image -- denoising strategy
    contractive: bool = False # choose if apply contraction to the loss of not
    lamb: float = 1e-3 # controls the relative importance of the Jacobian (contractive) loss.
    reduction: str = "mean" # "mean" or "sum" according to the reduction loss strategy
    slope: float = 0.5 # slope for the leaky relu in convolutions
    
    # MIXER stuff
    mixer_ae: bool = True # if you want to *treshold* with the mixer strategy or not
    dropout: float = 0.2 # dropout for the mixer classifier
    cross_w: float = 3 # the importance to give to the classification task wrt reconstruction one
    conv_channel: int = 1024 # number of latent channels conv if v2 (must be a power of 2)
    
    # LOGGING params
    log_images: int = 4 # how many images to log each time
    log_image_each_epoch: int = 2 # epochs interval we wait to log images   