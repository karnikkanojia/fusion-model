import torchxrayvision as xrv
import torchvision
import pytorch_lightning as pl
from argparse import Namespace
import sklearn
from torch.utils.data import DataLoader


class XRayDataModule(pl.LightningDataModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.data_dir = args.data_dir
        self.ratio = args.ratio
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.num_workers = args.num_workers
        self.args = args
        self.train_datasets, self.valid_datasets = self.setup("fit")

    def get_label_length(self):
        return len(self.train_datasets.pathologies)

    def get_transforms(self):
        transforms = torchvision.transforms.Compose(
            [
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(224),
            ]
        )

        if self.args.data_aug:
            data_aug = torchvision.transforms.Compose(
                [
                    xrv.datasets.ToPILImage(),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomAffine(
                        self.args.data_aug_rot,
                        translate=(self.args.data_aug_trans, self.args.data_aug_trans),
                        scale=(1.0 - self.args.data_aug_scale, 1.0 + self.args.data_aug_scale),
                    ),
                    torchvision.transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
                    torchvision.transforms.ToTensor(),
                ]
            )

            return transforms, data_aug

        return transforms, None

    def setup(self, stage: str) -> tuple[xrv.datasets.MergeDataset]:
        datas = []
        datas_names = []
        transforms, data_aug = self.get_transforms()
        if "nih" in self.args.dataset:
            dataset = xrv.datasets.NIH_Dataset(
                imgpath=self.args.data_dir + "/images-512-NIH",
                transform=transforms,
                data_aug=data_aug,
                unique_patients=False,
                views=["PA", "AP"],
            )
            datas.append(dataset)
            datas_names.append("nih")
        if "pc" in self.args.dataset:
            dataset = xrv.datasets.PC_Dataset(
                imgpath=self.args.data_dir + "/images-512-PC",
                transform=transforms,
                data_aug=data_aug,
                unique_patients=False,
                views=["PA", "AP"],
            )
            datas.append(dataset)
            datas_names.append("pc")
        if "chex" in self.args.dataset:
            dataset = xrv.datasets.CheX_Dataset(
                imgpath=self.args.data_dir + "/CheXpert-v1.0-small",
                csvpath=self.args.data_dir + "/CheXpert-v1.0-small/train.csv",
                transform=transforms,
                data_aug=data_aug,
                unique_patients=False,
            )
            datas.append(dataset)
            datas_names.append("chex")
        if "google" in self.args.dataset:
            dataset = xrv.datasets.NIH_Google_Dataset(
                imgpath=self.args.data_dir + "/images-512-NIH",
                transform=transforms,
                data_aug=data_aug,
            )
            datas.append(dataset)
            datas_names.append("google")
        if "mimic_ch" in self.args.dataset:
            dataset = xrv.datasets.MIMIC_Dataset(
                imgpath="/scratch/users/joecohen/data/MIMICCXR-2.0/files/",
                csvpath=self.args.data_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
                metacsvpath=self.args.data_dir
                + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
                transform=transforms,
                data_aug=data_aug,
                unique_patients=False,
                views=["PA", "AP"],
            )
            datas.append(dataset)
            datas_names.append("mimic_ch")
        if "openi" in self.args.dataset:
            dataset = xrv.datasets.Openi_Dataset(
                imgpath=self.args.data_dir + "/OpenI/images/",
                transform=transforms,
                data_aug=data_aug,
            )
            datas.append(dataset)
            datas_names.append("openi")
        if "rsna" in self.args.dataset:
            dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
                imgpath=self.args.data_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
                transform=transforms,
                data_aug=data_aug,
                unique_patients=False,
                views=["PA", "AP"],
            )
            datas.append(dataset)
            datas_names.append("rsna")
        if "siim" in self.args.dataset:
            dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
                imgpath=self.args.data_dir + "/SIIM_TRAIN_TEST/dicom-images-train",
                csvpath=self.args.data_dir + "/SIIM_TRAIN_TEST/train-rle.csv",
                transform=transforms,
                data_aug=data_aug,
            )
            datas.append(dataset)
            datas_names.append("siim")
        if "vin" in self.args.dataset:
            dataset = xrv.datasets.VinBrain_Dataset(
                imgpath=self.args.data_dir
                + "vinbigdata-chest-xray-abnormalities-detection/train",
                csvpath=self.args.data_dir
                + "vinbigdata-chest-xray-abnormalities-detection/train.csv",
                transform=transforms,
                data_aug=data_aug,
            )
            datas.append(dataset)
            datas_names.append("vin")

        train_datas = []
        test_datas = []
        for i, dataset in enumerate(datas):
            if "patientid" not in dataset.csv:
                dataset.csv["patientid"] = [
                    "{}-{}".format(dataset.__class__.__name__, i)
                    for i in range(len(dataset))
                ]

            gss = sklearn.model_selection.GroupShuffleSplit(
                train_size=0.8, test_size=0.2, random_state=self.args.seed
            )

            train_inds, test_inds = next(
                gss.split(X=range(len(dataset)), groups=dataset.csv.patientid)
            )
            train_dataset = xrv.datasets.SubsetDataset(dataset, train_inds)
            test_dataset = xrv.datasets.SubsetDataset(dataset, test_inds)

            train_datas.append(train_dataset)
            test_datas.append(test_dataset)

        if len(datas) == 0:
            raise Exception("No datasets found (check --dataset argument)")
        elif len(datas) == 1:
            train_dataset = train_datas[0]
            test_dataset = test_datas[0]
        else:
            print("Merging Dataset")
            train_dataset = xrv.datasets.Merge_Dataset(train_datas)
            test_dataset = xrv.datasets.Merge_Dataset(test_datas)

        return train_dataset, test_dataset

    def train_dataloader(self):
        # TODO: If you want to use custom sampler and loader follow like this
        # from transformers.trainer_pt_utils import DistributedLengthGroupedSampler
        # train_sampler = DistributedLengthGroupedSampler(
        # batch_size=self.per_device_eval_batch_size,
        # dataset=self.train_datasets,
        # model_input_name="features",
        # lengths=self.train_datasets["feature_lenghths"],
        # )
        # return CustomDataLoader(
        #     dataset=self.train_datasets,
        #     batch_size=self.per_device_train_batch_size,
        #     sampler=train_sampler,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        # )

        # TODO: If you want to use default loader
        return DataLoader(
            dataset=self.train_datasets,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_datasets,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        pass