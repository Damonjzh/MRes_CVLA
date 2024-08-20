import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import PIL.Image
from torchvision import transforms

def cohen_aug(img):
    # Follow https://arxiv.org/pdf/2002.02497.pdf, page 4
    # "Data augmentation was used to improve generalization.  According to best results inCohen et al. (2019) (and replicated by us)
    # each image was rotated up to 45 degrees, translatedup to 15% and scaled larger of smaller up to 10%"
    aug_ = A.Compose([
        A.ShiftScaleRotate(p=1.0, shift_limit=0.25, rotate_limit=45, scale_limit=0.1),
        A.HorizontalFlip(p=0.5),
    ])
    return aug_(image=img[0])["image"].reshape(img.shape)


class Mimic(Dataset):
    # def __init__(self, imgpath, csvpath, views=["PA"], transform=None, data_aug=None,
    # flat_dir=True, seed=0, unique_patients=True):
    def __init__(self, path, path2 = None, path3=None, split="train", type = "gt_finding", aug=None, transform=None, views=["AP", "PA"], unique_patients=False):
        super().__init__()
        if type == "gt_finding":
            if split == "train":
                csvpath = os.path.join(path, 'gt_finding_train_14label_binary.csv')
            elif split == "test":
                csvpath = os.path.join(path, 'gt_finding_test_14label_binary.csv')
            else:
                raise ValueError(csvpath)
        elif type == "r2gen":
            if split == "train":
                csvpath = os.path.join(path, 'r2gen_train_14labels_binary.csv')
            elif split == "val":
                csvpath = os.path.join(path, 'r2gen_val_14labels_binary.csv')
            elif split == "test":
                csvpath = os.path.join(path, 'r2gen_test_14labels_binary.csv')
            elif split == "all":
                csvpath = os.path.join(path, 'r2gen_all_14labels_binary.csv')
            else:
                raise ValueError(csvpath)

        elif type == "r2gen_cmn":
            if split == "train":
                csvpath = os.path.join(path, 'r2gencmn_train_14labels_binary.csv')
            elif split == "val":
                csvpath = os.path.join(path, 'r2gencmn_val_14labels_binary.csv')
            elif split == "test":
                csvpath = os.path.join(path, 'r2gencmn_test_14labels_binary.csv')
            elif split == "all":
                csvpath = os.path.join(path, 'r2gencmn_all_14labels_binary.csv')
            else:
                raise ValueError(csvpath)
        elif type == "rgrg":
            if split == "train":
                csvpath = os.path.join(path, 'rgrg_train_14labels_binary.csv')
            elif split == "val":
                csvpath = os.path.join(path, 'rgrg_val_14labels_binary.csv')
            elif split == "test":
                csvpath = os.path.join(path, 'rgrg_test_14labels_binary.csv')
            elif split == "all":
                csvpath = os.path.join(path, 'rgrg_all_14labels_binary.csv')
            else:
                raise ValueError(csvpath)
        elif type == "both":
            if split == "train":
                csvpath = os.path.join(path, 'gt_finding_train_14label_binary.csv')
                csvpath2 = os.path.join(path2, 'r2gen_train_14labels_binary.csv')
                csvpath3 = os.path.join(path3, 'r2gencmn_train_14labels_binary.csv')
                # csvpath2 = os.path.join(path2, 'rgrg_train_14labels_binary.csv')
            elif split == "val":
                csvpath = os.path.join(path, 'gt_finding_val_14label_binary.csv')
                csvpath2 = os.path.join(path2, 'r2gen_val_14labels_binary.csv')
                csvpath3 = os.path.join(path3, 'r2gencmn_val_14labels_binary.csv')
                # csvpath2 = os.path.join(path2, 'rgrg_val_14labels_binary.csv')
            elif split == "test":
                csvpath = os.path.join(path, 'gt_finding_test_14label_binary.csv')
                csvpath2 = os.path.join(path2, 'r2gen_test_14labels_binary.csv')
                csvpath3 = os.path.join(path3, 'r2gencmn_val_14labels_binary.csv')
                # csvpath2 = os.path.join(path2, 'rgrg_test_14labels_binary.csv')
            else:
                raise ValueError(csvpath)

        self.data_aug = aug
        # np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255

        self.pathologies = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
            "Support Devices"
        ]

        # self.pathologies = sorted(self.pathologies)

        self.imgpath = os.path.dirname(path)
        self.transform = transform
        self.data_aug = aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        self.image_paths = self.csv["image_path"].tolist()

        # To list
        # if type(self.views) is not list:
        #     views = [views]
        #     self.views = views

        # self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
        # self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv[
        #     "AP/PA"]  # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        # self.csv["view"] = self.csv["view"].replace({'Lateral': "L"})  # Rename Lateral with L
        # self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat='(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
        print(self.pathologies)

        ########## add consistent csv values

        # offset_day_int

        # patientid
        # if 'train' in csvpath:
        #     patientid = self.csv.Path.str.split("train/", expand=True)[1]
        # elif 'valid' in csvpath:
        #     patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        # elif 'test' in csvpath:
        #     patientid = self.csv["image_id"]
        # else:
        #     raise NotImplemented

        # patientid = patientid.str.split("/study", expand=True)[0]
        # patientid = patientid.str.replace("patient", "")
        self.csv["patientid"] = self.csv["image_id"]
        self.csv2 = None
        # self.gt_report = self.csv["report_gt"].tolist()
        # self.generated_report = self.csv["report_generated"].tolist()

        if type == "both":
            self.csv2 = pd.read_csv(csvpath2)
            self.image_paths2 = self.csv2["image_path"].tolist()
            self.labels2 = []
            for pathology in self.pathologies:
                if pathology in self.csv2.columns:
                    self.csv2.loc[healthy, pathology] = 0
                    mask = self.csv2[pathology]

                self.labels2.append(mask.values)
            self.labels2 = np.asarray(self.labels2).T
            self.labels2 = self.labels2.astype(np.float32)

            self.labels2[self.labels2 == -1] = np.nan
            self.csv2["patientid"] = self.csv2["image_id"]
            self.gt_report = self.csv2["report_gt"].tolist()
            self.generated_report = self.csv2["report_generated"].tolist()

            self.csv3 = pd.read_csv(csvpath3)
            self.image_paths3 = self.csv3["image_path"].tolist()
            self.labels3 = []
            for pathology in self.pathologies:
                if pathology in self.csv3.columns:
                    self.csv3.loc[healthy, pathology] = 0
                    mask = self.csv3[pathology]

                self.labels3.append(mask.values)
            self.labels3 = np.asarray(self.labels3).T
            self.labels3 = self.labels3.astype(np.float32)

            self.labels3[self.labels3 == -1] = np.nan
            self.csv3["patientid"] = self.csv3["image_id"]


    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(
            len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # imgid = self.csv['Path'].iloc[idx]

        img_path = self.image_paths[idx]
        gt_report = self.gt_report[idx]
        generated_report = self.generated_report[idx]


        img = Image.open(img_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")

        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512)
            ]
        )
        # img = image_transforms(img)


        # image = [img2]
        # w, h = image[0].size
        # w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8
        # image = [np.array(i.resize((w, h), resample=PIL.Image.LANCZOS))[None, :] for i in image]
        # image = np.concatenate(image, axis=0)
        # image = np.array(image).astype(np.float32) / 255.0
        # image = image.transpose(0, 3, 1, 2)
        # image = 2.0 * image - 1.0
        # img = torch.from_numpy(image[0])


        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)

        target = self.labels[idx]

        pathologies = []
        for i in range(len(target)):
            if target[i] == 1:
                pathologies.append(self.pathologies[i])
        if len(pathologies) > 0:
            pathologies = ",".join(pathologies)
        else:
            pathologies = ""
        target = torch.from_numpy(target).float()

        if self.csv2 is not None:
            target2 = self.labels2[idx]
            pathologies2 = []
            for i in range(len(target2)):
                if target2[i] == 1:
                    pathologies2.append(self.pathologies[i])
            if len(pathologies2) > 0:
                pathologies2 = ",".join(pathologies2)
            else:
                pathologies2 = ""

            target3 = self.labels3[idx]
            pathologies3 = []
            for i in range(len(target3)):
                if target3[i] == 1:
                    pathologies3.append(self.pathologies[i])
            if len(pathologies3) > 0:
                pathologies3 = ",".join(pathologies3)
            else:
                pathologies3 = ""

            # metadata = {'img_path': img_path, "pathologies": pathologies, "pathologies2": pathologies2, "gt_report": gt_report, "generated_report": generated_report}
            metadata = {'img_path': img_path, "pathologies": pathologies, "pathologies2": pathologies2, "pathologies3": pathologies3}
            # metadata = {'img_path': img_path, "pathologies": pathologies, "pathologies2": pathologies2}
            return img, target, target2, metadata
        else:
            # metadata = {'img_path': img_path, "pathologies": pathologies, "gt_report": gt_report, "generated_report": generated_report}
            metadata = {'img_path': img_path, "pathologies": pathologies}
            return img, target, metadata