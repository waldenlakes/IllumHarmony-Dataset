import os
import os.path as osp
import glob
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.utils as utils

from utils import extract_bounding_box, do_composition, randomly_choosing_a_point

class TrainIllumHarmonyDataset(Dataset):
    def __init__(self, dataset_dir, img_size = None):
        super(TrainIllumHarmonyDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.img_size = img_size

        self.background_images_dir = os.path.join(self.dataset_dir, 'background_imgs')
        # self.hdr_dir = os.path.join(self.dataset_dir, 'illum_maps')
        self.objects_dir = os.path.join(self.dataset_dir, 'objects')
        self.placement_masks_dir = os.path.join(self.dataset_dir, 'placement_binary_masks')

        self.gt_imgs_paths = []
        train_list_path = os.path.join(self.dataset_dir, "train_list.txt")
        for did in open(train_list_path):
            did = did.strip()
            self.gt_imgs_paths.append(osp.join(self.objects_dir, did))

    def __getitem__(self, index):
        # get image paths
        raw_gt_img_path = self.gt_imgs_paths[index]
        bg_img_path = self.get_corresponding_background(raw_gt_img_path)
        placemask_path = self.get_place_mask(bg_img_path)
        raw_mask_path = self.get_mask_foreground(raw_gt_img_path)
        raw_unharmonized_img_path = self.get_unharmonized_input(raw_gt_img_path, self.gt_imgs_paths) # RANDOMLY SELECTING

        # read images
        raw_gt_img = cv2.imread(raw_gt_img_path) / 255.0
        placemask = cv2.imread(placemask_path)[..., 0] / 255.0
        raw_mask = cv2.imread(raw_mask_path)[..., 0] / 255.0
        bg_img = cv2.imread(bg_img_path) / 255.0
        raw_unharmonized_img = cv2.imread(raw_unharmonized_img_path) / 255.0

        # randomly choosing a point in the placement mask
        placement_config = randomly_choosing_a_point(placemask)

        # object placement
        bb_gt_img, bb_mask = extract_bounding_box(raw_gt_img, raw_mask)
        gt_img, mask = do_composition(bb_gt_img, bb_mask, bg_img, placement_config)
        bb_unharmonized_img, bb_mask = extract_bounding_box(raw_unharmonized_img, raw_mask)
        unharmonized_img, mask = do_composition(bb_unharmonized_img, bb_mask, bg_img, placement_config)

        comp = np.flip(unharmonized_img, 2)
        gt = np.flip(gt_img, 2)
        mask = mask[..., 0:1]

        comp_tensor = torch.from_numpy(comp.copy()).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask.copy()).permute(2, 0, 1)

        if self.img_size is not None:
            comp_tensor = TF.resize(comp_tensor, [self.img_size, self.img_size])
            mask_tensor = TF.resize(mask_tensor, [self.img_size, self.img_size])
            gt_tensor = TF.resize(gt_tensor,[self.img_size, self.img_size])

        
        return {'comp': comp_tensor, 'mask': mask_tensor, 'gt': gt_tensor,'img_path':raw_gt_img_path}

    def _compose(self, foreground_img, foreground_mask, background_img):
        return foreground_img * foreground_mask + background_img * (1 - foreground_mask)

    def get_corresponding_background(self, indexed_image):

        image_name = indexed_image.split('/')[-1]
        scene_name = indexed_image.split('/')[-2]
        hdr_theta = image_name.split('_')[2].split('.')[0]

        return os.path.join(self.background_images_dir, f'{scene_name}_2.2_{hdr_theta}_0_90.png')

    def get_place_mask(self, indexed_background):
        th = indexed_background.split('/')[-1].split('_')[-3]
        ph = indexed_background.split('/')[-1].split('_')[-2]
        angle = indexed_background.split('/')[-1].split('_')[-1].split('.')[0]
        name = indexed_background.split('/')[-1].split('.')[0][:-2]

        return os.path.join(self.placement_masks_dir, name, f'background_2.2_{th}_{ph}_{angle}.png')

    def get_mask_foreground(self, indexed_image):
        ILLUM_NAME = indexed_image.split('/')[-4]
        OBJ_ANGLE = indexed_image.split('/')[-1].split('_')[1]

        return os.path.join(self.objects_dir, ILLUM_NAME, 'mask_foreground', f'image_{OBJ_ANGLE}.bmp')

    def get_unharmonized_input(self, indexed_image, images_gt_images_paths):

        # remove the indexed image ground truth from the list of all images
        new_paths = images_gt_images_paths.copy()
        new_paths.remove(indexed_image)

        # filter all the list such that only images that have the same object angle as that
        # of the ground truth image are retained
        unhamonized_inputs = []
        gt_angle = indexed_image.split('/')[-1].split('.')[0].split('_')[1]
        gt_name = indexed_image.split('/')[-4]

        for i in range(len(new_paths)):
            current_paths = new_paths[i]
            current_angle = current_paths.split('/')[-1].split('.')[0].split('_')[1]
            current_name = current_paths.split('/')[-4]

            if current_angle == gt_angle and current_name == gt_name:
                unhamonized_inputs.append(current_paths)

        # randomly select an image from the filtered list and return it
        unharmonized = unhamonized_inputs[random.randint(0, len(unhamonized_inputs) - 1)]
        return unharmonized

    def __len__(self):
        return len(self.gt_imgs_paths)


if __name__ == '__main__':
    # ------------------------- download the IllumHarmony-Dataset and set the directory of train set -------------------------
    TRAIN_DATA_DIR = "./datasets/IllumHarmonyDataset/train"

    # ------------------------- init dataloader-------------------------
    dataset = TrainIllumHarmonyDataset(TRAIN_DATA_DIR)
    print(f'Number of train images: {len(dataset)}')
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            num_workers=2,
                                            batch_size=1,
                                            shuffle=True)

    # ------------------------- load data -------------------------
    for iter, tensors_dic in enumerate(train_loader):
        iter = iter + 1
        print(f'Train [{iter}|{len(train_loader)}]')

        # load input composite image, mask and gt
        comp = tensors_dic['comp']
        mask = tensors_dic['mask']
        gt = tensors_dic['gt']

        # TODO: TRAIN YOUR OWN MODEL

