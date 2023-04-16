import torch
from .AE_simple import AE, conv_block
from torch import nn
import torch.nn.functional as F
from .data_module import MVTec_DataModule
from torchmetrics import F1Score
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import multiscale_structural_similarity_index_measure as MSSIM
from torchvision import models

# CLASSIFIER
class Obj_classifer(nn.Module):
	def __init__(self, ld, out_classes, hparams):
		# note that ld is the latent dim
		super().__init__()
		self.classifier = nn.Sequential(
			nn.Linear(ld, ld//2),
			nn.ReLU(inplace=True),
			nn.Dropout(hparams.dropout),

			nn.Linear(ld//2, ld//4),
			nn.ReLU(inplace=True),
			nn.Dropout(hparams.dropout),

			nn.Linear(ld//4, out_classes),
			nn.ReLU(inplace=True)
		)
	def forward(self, latent):
		return self.classifier(latent)

# CLASSIFIER with latent space Convolution
class Obj_classifer_conv(nn.Module):
	def __init__(self, ld, out_classes, hparams):
		# note that ld is the latent dim
		super().__init__()
		self.conv = nn.Sequential(*conv_block(hparams.conv_channel, ld, kernel_size=3, stride=1, padding=0, bias=False, slope = 0, normalize=True))
		self.classifier = nn.Sequential(
			nn.LazyLinear(ld),
			nn.ReLU(inplace=True),
			nn.Dropout(hparams.dropout),

			nn.LazyLinear(ld//2),
			nn.ReLU(inplace=True),
			nn.Dropout(hparams.dropout),

			nn.LazyLinear(out_classes),
			nn.ReLU(inplace=True)
		)
	def forward(self, latent):
		return self.classifier(self.conv(latent).view(latent.shape[0],-1))

class Mixer_AE(AE):
	def __init__(self, hparams):
		super(Mixer_AE, self).__init__(hparams)
		if self.hparams.version == "1":
			self.classifier = Obj_classifer(self.hparams.latent_size, self.hparams.obj_classes, self.hparams)
		elif self.hparams.version == "2":
			self.classifier = Obj_classifer_conv(self.hparams.latent_size, self.hparams.obj_classes, self.hparams)
		# we also tried an alternative experiment using resnet classifier, not from latent space but directly from the input image
		# self.classifier_res = Obj_classifer_resnet(self.hparams.obj_classes, self.hparams)
		
  		# if you want to predict using different tresholds you need to store 
		# different tresholds. If you prefer not then you can average all of them.
		# if we miss the tresholds param (not in a checkpoint, we load it)
		if not hasattr(self.hparams, 'thresholds'):
			self.hparams.thresholds = {a : self.hparams.threshold for a in MVTec_DataModule.id2c.keys()}
		# metric to log the classification problem
		self.val_f1score_classes = F1Score(task = 'multiclass', num_classes = self.hparams.obj_classes, average = 'macro')
	
	# in what follow we implement optionally the CONTRACTIVE and DENOISING behaviour	
	def forward(self, img):
		if self.hparams.noise > 0 and self.training:
			# we want to randomly add or remove
			img = img + (torch.rand_like(img)-torch.rand_like(img))*self.hparams.noise
			#we need to fix the values in the range [-1,1]
			img = img.clamp(min=-1, max=1)
		latent = self.encoder(img)
		return self.decoder(latent), self.classifier(latent)

	# here is encapsulated the prediction logic of the MIXER.
	def anomaly_prediction(self, img, recon = None, classes = None, batch = None):
		if recon is None:
			recon, classes = self(img)
		anomaly_score = self.anomaly_score(img, recon)
		if self.hparams.mixer_ae:
			classes = torch.argmax(classes, dim=-1).tolist() # these are the predicted classes
			threshold_idx = [self.hparams.thresholds[i] for i in classes] 
			ris = (anomaly_score > torch.tensor(threshold_idx, device = self.device)).long()
		# -- to test perfect predictions --
		# we used at the beginning for understanding if the mixer strategy would have been worth it or not
		# if self.hparams.mixer_ae:
		# 	classes = torch.argmax(classes, dim=-1).tolist() # these are the predicted classe
		# 	threshold_idx = [self.hparams.thresholds[i] for i in batch["class_obj"].tolist()] 
		# 	ris = (anomaly_score > torch.tensor(threshold_idx, device = self.device)).long()
		else: # we have also the possibility of not using the MIXER capability
			ris = (anomaly_score > self.hparams.threshold).long()
		return ris
	
	def loss_function(self,recon_x, batch_x, classes):
		loss_dict = dict()	
		loss = self.main_loss(recon_x, batch_x['img'])
		loss_dict["main_loss"] = loss
		if self.hparams.contractive:
			weights = torch.concat([param.view(-1) for param in self.encoder.parameters()])
			jacobian_loss = self.hparams.lamb*weights.norm(p='fro')
			loss += jacobian_loss
			loss_dict["jacobian_loss"] = jacobian_loss
		# here we compute the cross entropy loss for the classes, peculiarity of mixer AE
		cross_entropy_loss = self.hparams.cross_w*F.cross_entropy(classes, batch_x["class_obj"])
		loss_dict["cross_entropy_loss"] = cross_entropy_loss
		loss +=  cross_entropy_loss
		loss_dict["loss"] = loss
		return loss_dict

	def training_step(self, batch, batch_idx):
		imgs = batch['img']
		recon, classes = self(imgs)
		loss = self.loss_function(recon, batch, classes)
		# LOSS
		self.log_dict(loss)
		# ANOMALY SCORE --> mean and standard deviation for each class
		anomaly_scores = self.anomaly_score(imgs, recon).detach().cpu()
		all_std = dict()
		all_mean = dict()
		for k in self.hparams.thresholds.keys(): # for each class
			all_k = anomaly_scores[batch["class_obj"]==k]
			if all_k.nelement() == 0:
				# we skip if nothing to log
				continue
			all_mean[k] = all_k.mean().item()
			if all_k.nelement()>1:
				# std with only one element is not defined in pytorch (nan)
				all_std[k] = all_k.std().item()
		return {'loss': loss['loss'], 'anom': all_mean, 'a_std': all_std}

	def training_epoch_end(self, outputs):
		# we need to update all the thresholds
		all_tresh = list()
		for k in self.hparams.thresholds.keys(): # we update the thresholds for each class (not anymore for every category)
			a = np.array([x['anom'][k] for x in outputs if x['anom'].get(k,None) is not None]) 
			a_std = np.array([x['a_std'][k] for x in outputs if x['a_std'].get(k,None) is not None]) 
			avg_anomaly_k = a.mean()
			std_anomaly_k = a_std.mean()
			# if due to our standard deviation we have 0 std to compute, this values is simply 0
			# and could be manually finetuned later on.
			if std_anomaly_k == np.nan:
				std_anomaly_k = 0
			# THRESHOLD UPDATE
			self.hparams.thresholds[k] = (1-self.hparams.t_weight)*self.hparams.thresholds[k] + \
								self.hparams.t_weight*(avg_anomaly_k + self.hparams.w_std*std_anomaly_k)
			all_tresh.append(self.hparams.thresholds[k])
			self.log("anomaly_threshold."+MVTec_DataModule.id2c[k], self.hparams.thresholds[k], on_step=False, on_epoch=True, prog_bar=False)
		all_tresh = np.array(all_tresh)
		self.hparams.threshold = all_tresh.mean()
		self.log("anomaly_threshold_all_avg", self.hparams.threshold, on_step=False, on_epoch=True, prog_bar=True)
		
	def validation_step(self, batch, batch_idx):
		imgs = batch['img']
		recon_imgs, classes = self(imgs)
		# LOSS
		self.log("val_loss", self.loss_function(recon_imgs, batch, classes)["loss"], on_step=False, on_epoch=True, batch_size=imgs.shape[0])
		# RECALL, PRECISION, F1 on anomaly predicitons
		pred = self.anomaly_prediction(imgs, recon_imgs, classes=classes, batch=batch)
		self.val_precision.update(pred, batch['label'])
		self.val_recall.update(pred, batch['label'])
		self.val_f1score.update(pred, batch['label'])
		self.val_auroc.update(pred, batch['label'])
		self.log("precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("f1_score", self.val_f1score, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		self.log("auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=False, batch_size=imgs.shape[0])
		# F1 on classes predictions
		classes = torch.argmax(classes, dim = -1)
		self.val_f1score_classes.update(classes, batch["class_obj"])
		self.log("f1_score_classes", self.val_f1score_classes, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
		# IMAGES
		images = self.get_images_for_log(imgs[0:self.hparams.log_images], recon_imgs[0:self.hparams.log_images])
		return {"images": images}