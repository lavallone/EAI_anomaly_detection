import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from tqdm import tqdm

class MVTec_Dataset(Dataset):
	def __init__(self, dataset_dir: str, train_or_test: str, hparams):
		self.data = list() # list of images with their class and label (0 normal, 1 anomalous)
		self.train_or_test = train_or_test
		self.dataset_dir = dataset_dir
		self.hparams = hparams
		self.transform = transforms.Compose([
			transforms.Resize((hparams.img_size, hparams.img_size)),
			# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
			# to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
			transforms.ToTensor(),
			# to have 0 mean and values in range [-1, 1]
			# The parameters mean, std are passed as 0.5, 0.5 in your case. 
			# This will normalize the image in the range [-1,1]. For example,
			# the minimum value 0 will be converted to (0-0.5)/0.5=-1, 
			# the maximum value of 1 will be converted to (1-0.5)/0.5=1.
			# https://discuss.pytorch.org/t/understanding-transform-normalize/21730
			transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
		])
		if self.hparams.augmentation and self.train_or_test == "train": # a slightly image augmentation
				self.augmentation = transforms.Compose([transforms.RandomVerticalFlip(0.3), 
													 transforms.RandomHorizontalFlip(0.3)])		
		self.make_data()
	
	def make_data(self):
		# this function read the fresh downloaded dataset and make it ready for the training
		class_dir_list = list()
		for f in [os.path.join(self.dataset_dir, e) for e in os.listdir(self.dataset_dir)]:
			if os.path.isdir(f):
				class_dir_list.append(f)
		print(f"Loading {self.train_or_test} dataset...")
		for f in tqdm(class_dir_list, desc = f"{self.train_or_test} dataset", position=1, leave = True):
			class_obj = f.split("/")[-1]
			for dir in os.listdir(f):
				if dir==self.train_or_test:
					current_dir = os.path.join(f,dir)
					for t in os.listdir(current_dir):
						imgs = os.path.join(current_dir,t)
						label = 0 if t=="good" else 1
						for image_path in  tqdm([os.path.join(imgs,e) for e in os.listdir(imgs)], desc = class_obj, position=0, leave = False):
							img = self.transform(Image.open(image_path).convert('RGB'))
							self.data.append({"img" : img, "class_obj": MVTec_DataModule.c2id[class_obj], "label" : label})
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		# augmentation is performed only on the training set
		if self.hparams.augmentation and self.train_or_test == "train": # a slightly image augmentation
			data_tmp = self.data[idx]
			data_tmp["img"] = self.augmentation(data_tmp["img"])
			return data_tmp
		else:
			return self.data[idx]

class MVTec_DataModule(pl.LightningDataModule):
	# static objs
	c2id = {'hazelnut': 0, 'capsule': 1, 'pill': 2, 'tile': 3, 'screw': 4, 'wood': 5, 'zipper': 6, 
	'metal_nut': 7, 'transistor': 8, 'carpet': 9, 'bottle': 10, 'grid': 11, 'toothbrush': 12, 'leather': 13, 'cable': 14}
	id2c = {0: 'hazelnut', 1: 'capsule', 2: 'pill', 3: 'tile', 4: 'screw', 5: 'wood', 6: 'zipper', 
	7: 'metal_nut', 8: 'transistor', 9: 'carpet', 10: 'bottle', 11: 'grid', 12: 'toothbrush', 13: 'leather', 14: 'cable'}
	def __init__(self, hparams: dict):
		super().__init__()
		self.save_hyperparameters(hparams, logger=False)

	def setup(self, stage=None):
		if not hasattr(self,"data_train"):
			# TRAIN
			self.data_train = MVTec_Dataset(self.hparams.dataset_dir, "train", self.hparams)
			# TEST
			self.data_test = MVTec_Dataset(self.hparams.dataset_dir, "test", self.hparams)

	def train_dataloader(self):
		return DataLoader(
			self.data_train,
			batch_size=self.hparams.batch_size,
			shuffle=True,
			num_workers=self.hparams.n_cpu,
			pin_memory=self.hparams.pin_memory,
			persistent_workers=True
		)

	def val_dataloader(self):
		return DataLoader(
			self.data_test,
			batch_size=self.hparams.batch_size,
			shuffle=False,
			num_workers=self.hparams.n_cpu,
			pin_memory=self.hparams.pin_memory,
			persistent_workers=True
		)
	#to invert the normalization of the compose transform.
	@staticmethod
	def denormalize(tensor):
		return tensor*0.5 + 0.5