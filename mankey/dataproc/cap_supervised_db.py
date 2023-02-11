import json
import attr
import os
from typing import List
import yaml
import numpy as np
from mankey.utils.imgproc import PixelCoord, pixel_in_bbox
from mankey.utils.transformations import quaternion_matrix
from mankey.dataproc.supervised_keypoint_db import SupervisedImageKeypointDatabase, SupervisedKeypointDBEntry, sanity_check_spartan
import cv2
import tqdm
import matplotlib.pyplot as plt


@attr.s
class CapSupvervisedKeypointDBConfig:
    # ${pdc_data_root}/logs_proto/2018-10....
    data_dir = ''
    model_dir = ''
    
    # Output the loading process
    verbose = True


class CapSupervisedKeypointDatabase(SupervisedImageKeypointDatabase):
    """
    The spartan multi-view RGBD dataset with keypoint-annotation. This one
    is specified to one object and back-ground subtracted bounding box (instance mask).
    Compared with the tree-like config used in pytorch-dense-correspondence, I would
    favor simple, flatten dataset like this one ...
    """
    def __init__(self, config: CapSupvervisedKeypointDBConfig):
        super(CapSupervisedKeypointDatabase, self).__init__()
        self._config = config  # Not actually use it, but might be useful
        self._keypoint_entry_list: List[SupervisedKeypointDBEntry] = []
        self._num_keypoint = 4

        # For each scene
        scene_list = sorted(os.listdir(config.data_dir))
        scene_list = filter(lambda x: os.path.isdir(os.path.join(config.data_dir, x)), scene_list)
        scene_list = filter(lambda x: os.path.exists(os.path.join(config.data_dir, x, 'scene_camera.json')), scene_list)
        for scene_dir in tqdm.tqdm(scene_list):

            rgb_dir = os.path.join(config.data_dir, scene_dir, 'rgb')
            depth_dir = os.path.join(config.data_dir, scene_dir, 'depth')
            mask_visib_dir = os.path.join(config.data_dir, scene_dir, 'mask_visib')
            with open(os.path.join(config.data_dir, scene_dir, 'scene_camera.json'), 'r') as f:
                scene_camera = json.load(f)
            with open(os.path.join(config.data_dir, scene_dir, 'scene_gt.json'), 'r') as f:
                scene_gt = json.load(f)
            with open(os.path.join(config.data_dir, scene_dir, 'scene_gt_info.json'), 'r') as f:
                scene_gt_info = json.load(f)
            
            image_list = sorted(os.listdir(rgb_dir))
            for image_name in image_list:
                image_id = int(image_name.split('.')[0])
                for obj_id in range(2):
                    entry = SupervisedKeypointDBEntry()
                    
                    entry.rgb_image_path = os.path.join(rgb_dir, image_name)
                    entry.depth_image_path = os.path.join(depth_dir, image_name.replace('jpg', 'png'))
                    entry.binary_mask_path = os.path.join(mask_visib_dir, image_name.split('.')[0] + f'_{obj_id:06d}.png')
                    entry.bbox_top_left = PixelCoord()
                    entry.bbox_bottom_right = PixelCoord()
                    entry.bbox_top_left.x = scene_gt_info[str(image_id)][obj_id]['bbox_obj'][0]
                    entry.bbox_top_left.y = scene_gt_info[str(image_id)][obj_id]['bbox_obj'][1]
                    entry.bbox_bottom_right.x = scene_gt_info[str(image_id)][obj_id]['bbox_obj'][0] + scene_gt_info[str(image_id)][obj_id]['bbox_obj'][2]
                    entry.bbox_bottom_right.y = scene_gt_info[str(image_id)][obj_id]['bbox_obj'][1] + scene_gt_info[str(image_id)][obj_id]['bbox_obj'][3]
                    cam_R_m2c = np.array(scene_gt[str(image_id)][obj_id]['cam_R_m2c']).reshape(3, 3)
                    cam_t_m2c = np.array(scene_gt[str(image_id)][obj_id]['cam_t_m2c']).reshape(3, 1)
                    keypoint_obj = np.array([[25, 25, 25], [25, 25, -25], [-25, 25, 25], [-25, 25, -25],]).T # mm
                    entry.keypoint_camera = (cam_R_m2c @ keypoint_obj + cam_t_m2c) / 1000 # mm -> m
                    entry.camera_in_world = np.eye(4)
                    cam_K = np.array(scene_camera[str(image_id)]['cam_K']).reshape(3, 3)
                    keypoint_img = cam_K @ entry.keypoint_camera
                    keypoint_img = keypoint_img[:2, :] / keypoint_img[2, :]
                    keypoint_img = keypoint_img.T.astype(np.int32)
                    depth = cv2.imread(entry.depth_image_path, cv2.IMREAD_UNCHANGED)
                    depth = depth.astype(np.float32) * 0.1
                    keypoint_depth = depth[keypoint_img[:, 1], keypoint_img[:, 0]]
                    entry.keypoint_pixelxy_depth = np.concatenate([keypoint_img, keypoint_depth[:, np.newaxis]], axis=1).T.astype(int)
                    entry.keypoint_validity_weight = np.repeat((keypoint_img[:, 0] >= entry.bbox_top_left.x).astype(int) &
                                                               (keypoint_img[:, 0] <= entry.bbox_bottom_right.x).astype(int) &
                                                               (keypoint_img[:, 1] >= entry.bbox_top_left.y).astype(int) &
                                                               (keypoint_img[:, 1] <= entry.bbox_bottom_right.y).astype(int) &
                                                               (keypoint_depth > 0).astype(int), 3, 0).astype(np.float32).T
                    entry.on_boundary = np.min(entry.keypoint_validity_weight) == 0
                    self._keypoint_entry_list.append(entry)

        # Simple info
        print('The number of images is %d' % len(self._keypoint_entry_list))

    def get_entry_list(self) -> List[SupervisedKeypointDBEntry]:
        return self._keypoint_entry_list

    @property
    def num_keypoints(self):
        assert self._num_keypoint > 0
        return self._num_keypoint


# Simple code to test the db
def spartan_db_test():
    config = CapSupvervisedKeypointDBConfig()
    config.keypoint_yaml_name = 'shoe_6_keypoint_image.yaml'
    config.pdc_data_root = '/home/wei/data/pdc'
    config.config_file_path = '/home/wei/Coding/mankey/config/boot_logs.txt'
    database = CapSupervisedKeypointDatabase(config)
    entry_list = database.get_entry_list()
    for entry in entry_list:
        assert sanity_check_spartan(entry)


if __name__ == '__main__':
    spartan_db_test()
