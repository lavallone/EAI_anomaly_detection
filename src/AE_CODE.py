import torch
from .AE_simple import AE
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import multiscale_structural_similarity_index_measure as MSSIM

class CODE_AE(AE):
	def __init__(self, hparams):
		super(CODE_AE, self).__init__(hparams)
	def forward(self, img):
		# this implement the denoising autoencoder mechanism
		# here we preferred random whitenoise over zero-noise 
		if self.hparams.noise > 0:
			# we want to randomly add noise between -1 and 1
			img = img + (torch.rand_like(img)*2-1)*self.hparams.noise
			#we need to fix the values in the range [-1,1]
			img = img.clamp(min=-1, max=1)
		return self.decoder(self.encoder(img))

	def loss_function(self,recon_x, x):
		""" loss function is mse + Frobenius norm of 
		the jacobian matrix (of the Encoder w.r.t. to the inputs)   """
		loss_dict = dict()
		loss = self.main_loss(recon_x, x)
		loss_dict["main_loss"] = loss
		if self.hparams.contractive and self.training:
			# https://agustinus.kristia.de/techblog/2016/12/05/contractive-autoencoder/
			# https://github.com/AlexPasqua/Autoencoders
			# NOTE: the true implementation is what follows but it requires a lot of gpu memory,
			# (to store the full jacobian) since we are using only conv and relu this could be
			# approximated with frobenius norm of the encoder's weights as described in the paper
			# this is at all the effect a linearization of this regulation term
			# Compute the Jacobian of the encoder function with respect to the input data
			# jacobian = torch.autograd.functional.jacobian(self.encoder.convolutions, x, create_graph=False)[0]
			# Compute the Jacobian loss (fro = frobenius norm)
			# jacobian_loss = self.hparams.lambd * torch.norm(jacobian, p='fro', dim=(1,2))
			# jacobian_loss = jacobian_loss.mean()
			weights = torch.concat([param.view(-1) for param in self.encoder.parameters()])
			# as an alternative one could use only conv weights
			# weights_conv = [i.weight for i in self.encoder.convolutions if i.__class__.__name__.find("Conv2d")!= -1]
			jacobian_loss = self.hparams.lamb*weights.norm(p='fro')
			loss_dict["jacobian_loss"] = jacobian_loss
			loss += jacobian_loss
		loss_dict["loss"]  = loss
		return loss_dict