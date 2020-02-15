import collections
import os

from utils import training_utils

log = training_utils.get_logger()

GeneratorLosses = collections.namedtuple('GeneratorLosses', ['total_loss',
                                                             'confidence_reconstruction_loss',
                                                             'id_mrf_loss',
                                                             'global_wasserstein_loss',
                                                             'local_wasserstein_loss'])

DiscriminatorLosses = collections.namedtuple('DiscriminatorLosses', ['total_loss',
                                                                     'real_loss',
                                                                     'fake_loss',
                                                                     'gradient_penalty_loss'])


class WassersteinGAN:
  
  def __init__(self, img_height, img_width, num_channels, batch_size, n_critic, output_paths):
    """
    Abstract class representing Wasserstein GAN model framework. It is responsible for performing
    WGAN training procedure:
    1. Training critic in several steps (default 5),
    2. Training generator in one step.
    
    :param img_height: the input image height.
    :param img_width: the input image width.
    :param num_channels: the number of channels of input image.
    :param batch_size: the size of batch provided to the input models in each step.
    :param n_critic: the number of critic updates per generator update.
    """
    self.img_height = img_height
    self.img_width = img_width
    self.num_channels = num_channels
    self.batch_size = batch_size
    self.n_critic = n_critic
    self.output_paths = output_paths
    
    self.wgan_batch_size = self.n_critic * self.batch_size
    self.global_critic_weights_path = self.output_paths.global_critic_weights_path
    self.local_critic_weights_path = self.output_paths.local_critic_weights_file
    self.generator_weights_path = self.output_paths.generator_weights_path
  
  @property
  def global_discriminator(self):
    raise NotImplementedError
  
  @property
  def local_discriminator(self):
    raise NotImplementedError
  
  @property
  def generator(self):
    raise NotImplementedError
  
  @property
  def generator_for_prediction(self):
    raise NotImplementedError
  
  def train_global_discriminator(self, inputs, outputs):
    discriminator_loss = self.global_discriminator.train_on_batch(inputs, outputs)
    return discriminator_loss
  
  def train_local_discriminator(self, inputs, outputs):
    discriminator_loss = self.local_discriminator.train_on_batch(inputs, outputs)
    return discriminator_loss
  
  def train_generator(self, inputs, outputs):
    batch_g_inputs = self.get_batch(inputs, 0)
    batch_g_outputs = self.get_batch(outputs, 0)
    generator_loss = self.generator.train_on_batch(batch_g_inputs, batch_g_outputs)
    return generator_loss
  
  def predict(self, inputs):
    predicted_img = self.generator_for_prediction.predict_on_batch(inputs)
    return predicted_img
  
  def train_wgan(self, d_inputs, d_outputs, g_inputs, g_outputs):
    for i in range(self.n_critic):
      batch_d_inputs = self.get_batch(d_inputs, i)
      batch_d_outputs = self.get_batch(d_outputs, i)
      global_discriminator_loss = self.train_global_discriminator(batch_d_inputs, batch_d_outputs)
      local_discriminator_loss = self.train_local_discriminator(batch_d_inputs, batch_d_outputs)
    
    generator_loss = self.train_generator(g_inputs, g_outputs)
    
    generator_loss = GeneratorLosses(*generator_loss)
    global_discriminator_loss = DiscriminatorLosses(*global_discriminator_loss)
    local_discriminator_loss = DiscriminatorLosses(*local_discriminator_loss)
    return global_discriminator_loss, local_discriminator_loss, generator_loss
  
  def get_batch(self, wgan_batch, i):
    return [x[i * self.batch_size: (i + 1) * self.batch_size] for x in wgan_batch]
  
  def load(self, folder):
    log.info("Load weights from: %s", folder)
    self.generator.load_weights(os.path.join(folder, 'gmcnn.h5'), by_name=True, skip_mismatch=True)
    self.global_discriminator.load_weights(os.path.join(folder, 'global_critic.h5'), by_name=True,
                                           skip_mismatch=True)
    self.local_discriminator.load_weights(os.path.join(folder, 'local_critic.h5'), by_name=True,
                                          skip_mismatch=True)
    log.info('Loading done.')
  
  def save(self, folder):
      
    if os.path.exists(folder):
        log.info("Cant save weights! Folder exists: %s", folder)
        system.exit(-1)
        
    log.info("Saving weights to: %s", folder)
    os.makedirs(folder)
    self.generator.save_weights(os.path.join(folder, 'gmcnn.h5'), overwrite=True)
    self.global_discriminator.save_weights(os.path.join(folder, 'global_critic.h5'), overwrite=True)
    self.global_discriminator.save_weights(os.path.join(folder, 'local_critic.h5'), overwrite=True)
    log.info('Saving done.')
