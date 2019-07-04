import torch.utils.data as data
import os
from PIL import Image
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path, size=32):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        h,w = img.size
        img.crop(((w-size)/2, (h-size)/2, (w+size)/2, (h+size)/2))
        return img.convert('RGB')

class Alot(data.Dataset):

    base_folder = 'alot_grey/grey'

    def __init__(self, 
            root, 
            train=False, 
            extensions=IMG_EXTENSIONS, 
            loader=pil_loader,
            transform=None, 
            target_transform=None, 
            download=False,
            img_size=32):
        self.root = os.path.expanduser( root )
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.img_size = img_size

        images = []
        classes = []

        np.random.seed(31152)
        arr = np.arange(100)
        np.random.shuffle(arr)
        train_pos = arr[:75]
        test_pos = np.setdiff1d(np.arange(100), train_pos)

        if train:
            pos = train_pos
        else:
            pos = test_pos
        listdirs = os.listdir(os.path.join(self.root, self.base_folder))
        for dir in listdirs:
            listfiles = os.listdir(os.path.join(self.root, self.base_folder, dir))
            for t in pos:
                images.append(os.path.join(self.root, self.base_folder, dir, listfiles[t]))
                classes.append(dir)
        
        classes_name = np.unique(classes)
        class_to_idx = { classes_name[i]: i for i in range(len(classes_name))}

        samples=[]
        for i,c in zip(images, classes):
            samples.append((i,class_to_idx[c]))

        self.extensions = extensions 
        self.classes = classes_name 
        self.class_to_idx = class_to_idx
        self.samples = samples 
        self.targets = np.array([s[1] for s in samples])
    
    def __getitem__(self, idx):
        """
        Args:
        idx (int): Index

        Returns:
        tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[ idx  ]
        sample = self.loader(path, self.img_size)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

class AlotMetricLearning(Alot):
    num_training_classes = 250

    def __init__(self, 
            root, 
            train=False, 
            extensions=IMG_EXTENSIONS, 
            loader=pil_loader,
            transform=None, 
            target_transform=None, 
            download=False, 
            img_size=32
        ):
        self.root = os.path.expanduser( root )
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        images = []
        classes = []

        listdirs = os.listdir(os.path.join(self.root, self.base_folder))
        for dir in listdirs:
            listfiles = os.listdir(os.path.join(self.root, self.base_folder, dir))
            for f in listfiles:
                images.append(os.path.join(self.root, self.base_folder, dir, f))
                classes.append(dir)
        
        classes_name = np.unique(classes)
        class_to_idx = { classes_name[i]: i for i in range(len(classes_name))}

        samples=[]
        for i,c in zip(images, classes):
            samples.append((i,class_to_idx[c]))

        self.extensions = extensions 
        self.classes = classes_name 
        self.class_to_idx = class_to_idx
        self.samples = samples 
        self.targets = np.array([s[1] for s in samples])
    