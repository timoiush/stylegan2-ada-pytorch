import os
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy
import torch
import pandas as pd


"""
Figure functions
"""
def show_dataset(dataset, n_imgs=10, start=0, step=1, from_seed=False, with_slc=False):
    """
    Args:
        dataset (str):  directory containing images
        n_imgs (int): number of images to show
        start (int): start index (default: 0)
        step (int): step to iterate over image files (default: 1) 
        from_seed (bool): if the given dataset contains images from generate.py with title 'seed*'  
        with_slc (bool): if the file name contains slice position info   
    """
    imgs = [os.path.join(dataset, os.listdir(dataset)[start + i * step]) for i in range(0, n_imgs)]
    plt.figure(figsize=(2*n_imgs, 2))
    for i in range(0, n_imgs):
        ax = plt.subplot(1, n_imgs, i+1)
        img = plt.imread(imgs[i])
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if from_seed:
            title = os.path.basename(imgs[i]).split('.')[0]
            ax.set_title(title)
        if with_slc:
            slc = os.path.basename(imgs[i]).split('.')[0].split('-slice')[1]
            ax.set_title(f'slice {slc}')
    plt.show()


def extract_fid(fdir, log='metric-fid50k_full.jsonl'):
    '''
    Args:
        fdir (str): path of training-runs
        log (str): metric log file    
    '''
    data = [json.loads(line) for line in open(os.path.join(fdir, log), 'r')]
    fids = [item['results']['fid50k_full'] for item in data]

    return fids


def show_fid(fids, snap=20, log_file=None):
    """
    Args:
        fids (list): list containing fid scores in float
        log_file (str): path to log.txt of StaleGAN training; to get training time 
        snap (int): snap or tick specified when training the StyleGAN. fid score is calculated for each snap.
    """
    kimgs = np.arange(0, 4*snap*len(fids), 4*snap)  # one snap (tick) = 4 kimg
    if log_file:
        with open(log_file) as f:
            lines = f.readlines()[-5:]
            for i in range(-1, -6, -1):
                if not lines[i].startswith('tick'):
                    continue
                else:
                    break
            line = lines[i]            
            training_time = line.split('  ')[2].split('time ')[1]
    x_min = 4 * snap * np.argmin(fids)
    y_min = np.min(fids)

    plt.figure(figsize=(15, 5))
    plt.plot(kimgs, fids, label='FID score')
    plt.title(f'Training result: after {training_time}', fontsize=15)
    plt.xlabel('kimg (1000 images)', fontsize=15)
    plt.ylabel('FID score', fontsize=15)
    plt.annotate(f'min FID: {y_min:.4f}, \n{x_min} kimg', xy=(x_min, y_min), xytext=(x_min, y_min + 100), \
            arrowprops=dict(facecolor='black', shrink=0.05))
    plt.legend()
    plt.show()


def generate_images(gen_file, network_pkl, seeds, outdir, trunc=1.0):
    """
    Args: 
        gen_file (str): path of generate.py of StyleGAN
        network_pkl (str): path of StyleGAN model pkl file to generate image
        seeds (str): range of seeds
        outdir (str): output directory
        trunc (float): truncation rate of StyleGAN(default: 1.0)
    """
    
    os.system(f'python {gen_file} --outdir={outdir} --trunc={trunc} --seeds={seeds} --network={network_pkl}')
    show_dataset(outdir, n_imgs=3, from_seed=True)


def plot3planes(vol, ax_aspect=1, sag_aspect=1, cor_aspect=1):
    """ Plot the three planes (axial, sagittal, coronal) of volumetric data.
    Args:
        vol (numpy array): 3D volumetric data
        ax_aspect (float): aspect ratio in axial plane (x-y)
        sag_aspect (float): aspect ratio in sagittal plane (y-z)
        cor_aspect (float): aspect ratio in coronal plane (x-z)
    """
    vol_shape = vol.shape
    assert len(vol_shape) == 3, f'Volume has only {len(img_shape)} dimensions.'
    plt.figure(figsize=(10, 4))
    a1 = plt.subplot(1, 3, 1)
    plt.imshow(vol[:, :, vol_shape[2]//2], cmap='gray')
    a1.set_aspect(ax_aspect)
    a1.set_title('Axial')

    a2 = plt.subplot(1, 3, 2)
    img_s = vol[:, vol_shape[1]//2, :]
    img_s = scipy.ndimage.rotate(img_s, 90)
    plt.imshow(img_s, cmap='gray')
    a2.set_aspect(sag_aspect)
    a2.set_title('Sagittal')

    a3 = plt.subplot(1, 3, 3)
    plt.imshow(vol[vol_shape[0]//2, :, ::-1].T, cmap='gray')
    a3.set_aspect(cor_aspect)
    a3.set_title('Coronal');

def plot_slices(vol):
    """ Plot slices from volumetric data.
    Args:
        vol (numpy array): 3D volumetric data [h, w, c]
    """
    assert len(vol.shape) == 3, f'Volume has only {len(img_shape)} dimensions.'
    n_slc = vol.shape[2]
    n_rows = n_slc // 10 + 1
    fig, ax = plt.subplots(n_rows, 10, figsize=(15, 1.6 * n_rows), sharex=True)
    for i in range(10 * n_rows):
        plt_x = i % 10
        plt_y = i // 10
        if i < n_slc:
            ax[plt_y, plt_x].imshow(vol[:, :, i], cmap='gray')
            ax[plt_y, plt_x].set_title(f'index {i}')
            ax[plt_y, plt_x].axis('off')
        else:
            fig.delaxes(ax[plt_y, plt_x])
    plt.show() 


"""
Noise analysis
"""
def plot_noise_strength(network_dir, resolution=256, snap=20):
    '''
    Args:
        network_dir (str): directory of network pkl files
        resolution (int)
        snap (str): tick that saves model during training    
    '''
    blocks = []
    layers = []
    networks = glob.glob(os.path.join(network_dir, '*.pkl'))
    for i in range(2, int(np.log2(resolution))):
        block = f'b{2**i}'
        blocks.append(block)
        if block != 'b4':
            layers.append(f'{block}.conv0') 
        layers.append(f'{block}.conv1') 
            
            
    noise_scales = {layer:[] for layer in layers} 
    
    # Iterate over networks tp get noise
    for i, n in enumerate(networks):
        with open(n, 'rb') as f:
            G = pickle.load(f)['G_ema']
        na = NoiseAnalyzer(G)
        noise = na.get_noise()
        for layer in layers:
            noise_scales[layer].append(noise.loc[layer, 'noise_strength'])
            
    # Plot noise scales
    n = len(blocks)
    kimgs = np.arange(0, 4*snap*len(networks), 4*snap) 
    plt.figure(figsize=(15, 5))
    
    for i, layer in enumerate(layers):
        res = block.split('b')[1]
        plt.plot(kimgs, noise_scales[layer], label=layer)
    plt.title('Noise strength across conv layers', fontsize=15)
    plt.xlabel('kimg (1000 images)', fontsize=15)
    plt.ylabel('Noise strength', fontsize=15)
    plt.axis([0, kimgs.max()+1000, -0.12, 0.12])
    plt.legend()
    plt.show()


class NoiseAnalyzer:
    '''
        NoiseAnalyzer object to get the noise information of StyleGAN synthesis network (G.synthesis).
    '''
    def __init__(self, G):
        self.G = G
        self.synthesis = G.synthesis
        self.resolution = G.img_resolution
        self.noise = {}
        self.blocks = []
        
        for i in range(2, int(np.log2(self.resolution))):
            self.blocks.append(f'b{2**i}')
            
    def get_noise(self, mode='const'):
        ''' Get noise content from pre-trained model
        Args:
            mode (str): 'const' or 'random'
        Return:
            noise_df (DataFrame): 'noise_strength', 'const_noise' (only for 'const_noise')
        '''
        noise_norms = {}    
        for block in self.blocks:
            if block != 'b4':
                layer = f'{block}.conv0'
                self.noise[layer] = getattr(self.synthesis, block).conv0.noise_strength.data.numpy()
                if mode == 'const': 
                    noise_norms[layer] = torch.norm(getattr(self.synthesis, block).conv0.noise_const).numpy()
            layer = f'{block}.conv1'
            self.noise[layer] = getattr(self.synthesis, block).conv1.noise_strength.data.numpy()
            if mode == 'const': 
                noise_norms[layer] = torch.norm(getattr(self.synthesis, block).conv1.noise_const).numpy()
        noise_df = pd.DataFrame([self.noise, noise_norms]).T
        noise_df = noise_df.rename(columns={0: 'noise_strength', 1: 'const_noise'})
        return noise_df
    
    def plot_noise(self):
        ''' Visualize noise from seynthesis network layer '''
        n = len(self.blocks)
        fig, ax = plt.subplots(2, n, figsize=(n*4, 8))  # set real size: sharex=True, sharey=True
        for i, block in enumerate(self.blocks):
            res = block.split('b')[1]
            conv1_array = getattr(self.synthesis, block).conv1.noise_const.data
            ax[1, i].imshow(conv1_array, cmap='gray')
            ax[1, i].set_title(f'conv1: {res}x{res}')
            ax[1, i].axis('off')
            if block != 'b4':
                conv0_array = getattr(self.synthesis, block).conv0.noise_const.data
                ax[0, i].imshow(conv0_array, cmap='gray')
                ax[0, i].set_title(f'conv0: {res}x{res}')
                ax[0, i].axis('off')
        plt.show()


"""
Others
"""

def load_latent(latent_dir):
    latent = np.load(latent_dir)['w']
    return latent


def get_projection(img_path, network_pkl, outdir='tmp', ind=0, n_steps=1000):
    ''' Compute latent code of an image given a pretrained StyleGAN model.
    Args:
        img_path (str): target image path
        network_pkl (str): network path
        outdir (str): directory to save projected latent code (default: 'tmp')
        ind (int):
        n_steps (int): number of optimization steps (default: 1000)
    Return:
        w (numpy array): latent code in w space with dimension (512)
    '''
    path = '/data/projects/ml-ms-investigation/scripts/stylegan2-ada-pytorch'
    os.chdir(path)
    os.system(f'python3 projector.py --outdir={outdir} --target={img_path} --network={network_pkl} \
                                    --index={ind} --num-steps={n_steps}')
    w = load_latent(os.path.join(outdir, 'projected_w-0.npz'))
    return w[0, 0, :]



