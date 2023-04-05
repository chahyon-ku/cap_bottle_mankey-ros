import json
import shutil
import typing
import torch
import os
import cv2
import numpy as np
import random
from torch.utils.data import DataLoader
import tqdm
from mankey.network.resnet_nostage import ResnetNoStageConfig, ResnetNoStage, init_from_modelzoo
from mankey.network.weighted_loss import weighted_mse_loss, weighted_l1_loss
import mankey.network.predict as predict
import mankey.config.parameter as parameter
from mankey.dataproc.cap_supervised_db import CapSupvervisedKeypointDBConfig, CapSupervisedKeypointDatabase
from mankey.dataproc.supervised_keypoint_loader import SupervisedKeypointDatasetConfig, SupervisedKeypointDataset


# Some global parameter
learning_rate = 2e-4
n_epoch = 120
heatmap_loss_weight = 0.1


def construct_dataset(is_train: bool) -> typing.Tuple[SupervisedKeypointDataset, SupervisedKeypointDatasetConfig]:
    # Construct the db info
    db_config = CapSupvervisedKeypointDBConfig()
    db_config.data_dir = '/home/rpm/Lab/cap_bottle/data_generation/datasets/train_10k' if is_train else '/home/rpm/Lab/cap_bottle/data_generation/datasets/test_1k'
    db_config.model_dir = '/home/rpm/Lab/cap_bottle/data_generation/models'
    db_config.verbose = True

    # Construct the database
    database = CapSupervisedKeypointDatabase(db_config)

    # Construct torch dataset
    config = SupervisedKeypointDatasetConfig()
    config.network_in_patch_width = 256
    config.network_in_patch_height = 256
    config.network_out_map_width = 64
    config.network_out_map_height = 64
    config.image_database_list.append(database)
    config.is_train = is_train
    config.depth_scale = 0.1
    dataset = SupervisedKeypointDataset(config)
    return dataset, config


def construct_network():
    net_config = ResnetNoStageConfig()
    net_config.num_keypoints = 4
    net_config.image_channels = 4
    net_config.depth_per_keypoint = 3  # For integral heatmap, depthmap and regress heatmap
    net_config.num_layers = 34
    network = ResnetNoStage(net_config)
    return network, net_config


def visualize_entry(
        entry_idx: int,
        network: torch.nn.Module,
        dataset: SupervisedKeypointDataset,
        config: SupervisedKeypointDatasetConfig,
        save_dir: str):
    # The raw input
    processed_entry = dataset.get_processed_entry(dataset.entry_list[entry_idx])

    # The processed input
    stacked_rgbd = dataset[entry_idx]['rgbd_image']
    stacked_rgbd = torch.from_numpy(stacked_rgbd)
    stacked_rgbd = torch.unsqueeze(stacked_rgbd, dim=0)
    stacked_rgbd = stacked_rgbd.to('cuda:1')

    # Do forward
    raw_pred = network(stacked_rgbd)
    prob_pred = raw_pred[:, 0:dataset.num_keypoints, :, :]
    depthmap_pred = raw_pred[:, dataset.num_keypoints:2*dataset.num_keypoints, :, :]
    regress_heatmap = raw_pred[:, 2*dataset.num_keypoints:, :, :]
    heatmap = predict.heatmap_from_predict(prob_pred, dataset.num_keypoints)
    coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, dataset.num_keypoints)
    depth_pred = predict.depth_integration(heatmap, depthmap_pred)

    # To actual image coord
    coord_x = coord_x.cpu().detach().numpy()
    coord_y = coord_y.cpu().detach().numpy()
    coord_x = (coord_x + 0.5) * config.network_in_patch_width
    coord_y = (coord_y + 0.5) * config.network_in_patch_height

    # To actual depth value
    depth_pred = depth_pred.cpu().detach().numpy()
    depth_pred = (depth_pred * config.depth_image_scale) + config.depth_image_mean

    # Combine them
    keypointxy_depth_pred = np.zeros((3, dataset.num_keypoints), dtype=int)
    keypointxy_depth_pred[0, :] = coord_x[0, :, 0].astype(int)
    keypointxy_depth_pred[1, :] = coord_y[0, :, 0].astype(int)
    keypointxy_depth_pred[2, :] = depth_pred[0, :, 0].astype(int)

    # Get the image
    from mankey.utils.imgproc import draw_image_keypoint, draw_visible_heatmap
    # keypoint_rgb_cv = draw_image_keypoint(processed_entry.cropped_rgb, keypointxy_depth_pred, processed_entry.keypoint_validity)
    print(processed_entry.keypoint_xy_depth[0, 0], processed_entry.keypoint_xy_depth[0, 1])
    gt_keypoint_xy = processed_entry.keypoint_xy_depth[0:2, :].astype(int)
    print(gt_keypoint_xy.shape)
    keypoint_rgb_cv = cv2.circle(processed_entry.cropped_rgb, (gt_keypoint_xy[0, 0], gt_keypoint_xy[1, 0]), 1, (0, 0, 255), -1)
    keypoint_rgb_cv = cv2.circle(keypoint_rgb_cv, (gt_keypoint_xy[0, 1], gt_keypoint_xy[1, 1]), 1, (0, 255, 0), -1)
    keypoint_rgb_cv = cv2.circle(keypoint_rgb_cv, (gt_keypoint_xy[0, 2], gt_keypoint_xy[1, 2]), 1, (0, 0, 0), -1)
    keypoint_rgb_cv = cv2.circle(keypoint_rgb_cv, (gt_keypoint_xy[0, 3], gt_keypoint_xy[1, 3]), 1, (255, 0, 0), -1)

    keypoint_rgb_cv = cv2.circle(keypoint_rgb_cv, (keypointxy_depth_pred[0, 0], keypointxy_depth_pred[1, 0]), 1, (255, 0, 255), -1)
    keypoint_rgb_cv = cv2.circle(keypoint_rgb_cv, (keypointxy_depth_pred[0, 1], keypointxy_depth_pred[1, 1]), 1, (0, 255, 255), -1)
    keypoint_rgb_cv = cv2.circle(keypoint_rgb_cv, (keypointxy_depth_pred[0, 2], keypointxy_depth_pred[1, 2]), 1, (128, 128, 128), -1)
    keypoint_rgb_cv = cv2.circle(keypoint_rgb_cv, (keypointxy_depth_pred[0, 3], keypointxy_depth_pred[1, 3]), 1, (255, 255, 0), -1)
    rgb_save_path = os.path.join(save_dir, 'image_%d_rgb.png' % entry_idx)
    if os.path.exists(rgb_save_path):
        os.remove(rgb_save_path)
    cv2.imwrite(rgb_save_path, keypoint_rgb_cv)

    # The depth error
    depth_error_mm = np.abs(processed_entry.keypoint_xy_depth[2, :] - keypointxy_depth_pred[2, :])
    max_depth_error = np.max(depth_error_mm)
    print('Entry %d' % entry_idx)
    print('The max depth error (mm) is ', max_depth_error)

    # The pixel error
    pixel_error = np.sum(np.sqrt((processed_entry.keypoint_xy_depth[0:2, :] - keypointxy_depth_pred[0:2, :])**2), axis=0)
    max_pixel_error = np.max(pixel_error)
    print('The max pixel error (pixel in 256x256 image) is ', max_pixel_error)

    # Save the heatmap for the one with largest pixel error
    max_error_keypoint = np.argmax(pixel_error)
    raw_heatmap_np = regress_heatmap.cpu().detach().numpy()[0, max_error_keypoint, :, :]
    heatmap_vis = draw_visible_heatmap(raw_heatmap_np)
    heatmap_save_path = os.path.join(save_dir, 'image_%d_heatmap.png' % entry_idx)
    if os.path.exists(heatmap_save_path):
        os.remove(heatmap_save_path)
    cv2.imwrite(heatmap_save_path, heatmap_vis)


def visualize(network_path: str, save_dir: str):
    # Get the network
    network, _ = construct_network()

    # Load the network
    network.load_state_dict(torch.load(network_path))
    network.to('cuda:1')
    network.eval()

    # Construct the dataset
    dataset, config = construct_dataset(is_train=False)

    # try the entry
    num_entry = 50
    entry_idx = []
    for i in range(num_entry):
        entry_idx.append(random.randint(0, len(dataset) - 1))

    # A good example and a bad one
    for i in range(len(entry_idx)):
        visualize_entry(entry_idx[i], network, dataset, config, save_dir)


def evaluate(network_path: str, save_path: str):
    # Get the network
    network, _ = construct_network()

    # Load the network
    network.load_state_dict(torch.load(network_path))
    network.to('cuda:1')
    network.eval()

    # Construct the dataset
    dataset, config = construct_dataset(is_train=False)

    # A good example and a bad one
    results = {}
    for entry_idx in tqdm.tqdm(range(len(dataset))):
        # The raw input
        processed_entry = dataset.get_processed_entry(dataset.entry_list[entry_idx])

        # The processed input
        stacked_rgbd = dataset[entry_idx]['rgbd_image']
        stacked_rgbd = torch.from_numpy(stacked_rgbd)
        stacked_rgbd = torch.unsqueeze(stacked_rgbd, dim=0)
        stacked_rgbd = stacked_rgbd.to('cuda:1')

        # Do forward
        raw_pred = network(stacked_rgbd)
        prob_pred = raw_pred[:, 0:dataset.num_keypoints, :, :]
        depthmap_pred = raw_pred[:, dataset.num_keypoints:2*dataset.num_keypoints, :, :]
        regress_heatmap = raw_pred[:, 2*dataset.num_keypoints:, :, :]
        heatmap = predict.heatmap_from_predict(prob_pred, dataset.num_keypoints)
        coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(heatmap, dataset.num_keypoints)
        depth_pred = predict.depth_integration(heatmap, depthmap_pred)

        # To actual image coord
        coord_x = coord_x.cpu().detach().numpy()
        coord_y = coord_y.cpu().detach().numpy()
        coord_u = (coord_x + 0.5) * config.network_in_patch_width
        coord_v = (coord_y + 0.5) * config.network_in_patch_height
        coord_uv1 = np.concatenate((coord_u, coord_v, np.ones((1, dataset.num_keypoints, 1))), axis=0)
        bbox2patch = np.concatenate((processed_entry.bbox2patch, np.array([[0, 0, 1]])), axis=0)
        # print(np.linalg.inv(bbox2patch))
        img_uv1 = np.linalg.inv(bbox2patch) @ coord_uv1[:, :, 0]
        img_xy1 = np.linalg.inv(dataset.entry_list[entry_idx].cam_K) @ img_uv1

        # To actual depth value
        depth_pred = depth_pred.cpu().detach().numpy()
        depth_pred = (depth_pred * config.depth_image_scale) + config.depth_image_mean

        # ICP on keypoints
        gt_keypoints = 30 * np.array([[1, 1, 1], [1, 1, -1], [-1, 1, 1], [-1, 1, -1],]).T # mm
        gt_keypoints[1, :] += 10
        gt_keypoint_center = np.mean(gt_keypoints, axis=1)
        gt_centered_keypoints = gt_keypoints - np.expand_dims(gt_keypoint_center, axis=1)

        # Combine them
        # pred_keypoints = img_xy1 * processed_entry.keypoint_xy_depth[2, :]#depth_pred[0, :, 0]
        pred_keypoints = img_xy1 * depth_pred[0, :, 0]
        # print(pred_keypoints)
        pred_keypoint_center = np.mean(pred_keypoints, axis=1)
        pred_centered_keypoints = pred_keypoints - np.expand_dims(pred_keypoint_center, axis=1)
        
        H = gt_centered_keypoints @ pred_centered_keypoints.T
        U, X, V = np.linalg.svd(H)
        R = V.T @ U.T
        if np.linalg.det(R) < 0:
            V[2, :] *= -1
            R = V.T @ U.T
        assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

        t = pred_keypoint_center - R @ gt_keypoint_center

        sample_idx = int(dataset.entry_list[entry_idx].binary_mask_path.split('/')[-3])
        frame_idx = int(dataset.entry_list[entry_idx].binary_mask_path.split('/')[-1].split('_')[0])
        obj_idx = int(dataset.entry_list[entry_idx].binary_mask_path.split('/')[-1].split('_')[1].split('.')[0])
        if f'{sample_idx}/{frame_idx}' not in results:
            results[f'{sample_idx}/{frame_idx}'] = []
        results[f'{sample_idx}/{frame_idx}'].insert(obj_idx, {'cam_R_m2c': R.tolist(), 'cam_t_m2c': t.tolist(), 'obj_id': obj_idx + 1})
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=1)



def train(checkpoint_dir: str, start_from_ckpnt: str = '', save_epoch_offset: int = 0):
    # Construct the dataset
    dataset_train, train_config = construct_dataset(is_train=True)

    # And the dataloader
    loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=4)

    # Construct the regressor
    network, net_config = construct_network()
    if len(start_from_ckpnt) > 0:
        network.load_state_dict(torch.load(start_from_ckpnt))
    else:
        init_from_modelzoo(network, net_config)
    network.to('cuda:1')

    # The checkpoint
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # The optimizer and scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 90], gamma=0.1)

    # The loss for heatmap
    heatmap_criterion = torch.nn.MSELoss().to('cuda:1')

    # The training loop
    for epoch in range(n_epoch):
        # Save the network
        if epoch % 4 == 0 and epoch > 0:
            file_name = 'checkpoint-%d.pth' % (epoch + save_epoch_offset)
            checkpoint_path = os.path.join(checkpoint_dir, file_name)
            print('Save the network at %s' % checkpoint_path)
            torch.save(network.state_dict(), checkpoint_path)

        # Prepare info for training
        network.train()
        train_error_xy = 0
        train_error_depth = 0
        train_error_xy_heatmap = 0

        # The learning rate step
        scheduler.step()
        for param_group in optimizer.param_groups:
            print('The learning rate is ', param_group['lr'])

        # The training iteration over the dataset
        for idx, data in enumerate(loader_train):
            # Get the data
            image = data[parameter.rgbd_image_key]
            keypoint_xy_depth = data[parameter.keypoint_xyd_key]
            keypoint_weight = data[parameter.keypoint_validity_key]
            target_heatmap = data[parameter.target_heatmap_key]

            # Upload to cuda
            image = image.to('cuda:1')
            keypoint_xy_depth = keypoint_xy_depth.to('cuda:1')
            keypoint_weight = keypoint_weight.to('cuda:1')
            target_heatmap = target_heatmap.to('cuda:1')

            # To predict
            optimizer.zero_grad()
            raw_pred = network(image)
            prob_pred = raw_pred[:, 0:net_config.num_keypoints, :, :]
            depthmap_pred = raw_pred[:, net_config.num_keypoints:2*net_config.num_keypoints, :, :]
            regress_heatmap = raw_pred[:, 2*net_config.num_keypoints:, :, :]
            integral_heatmap = predict.heatmap_from_predict(prob_pred, net_config.num_keypoints)
            _, _, heatmap_height, heatmap_width = integral_heatmap.shape

            # Compute the coordinate
            coord_x, coord_y = predict.heatmap2d_to_normalized_imgcoord_gpu(integral_heatmap, net_config.num_keypoints)
            depth_pred = predict.depth_integration(integral_heatmap, depthmap_pred)

            # Concantate them
            xy_depth_pred = torch.cat((coord_x, coord_y, depth_pred), dim=2)

            # Compute loss
            loss = weighted_mse_loss(xy_depth_pred, keypoint_xy_depth, keypoint_weight)
            loss = loss + heatmap_loss_weight * heatmap_criterion(regress_heatmap, target_heatmap)

            # Do update
            loss.backward()
            optimizer.step()

            # Log info
            xy_error = float(weighted_l1_loss(xy_depth_pred[:, :, 0:2], keypoint_xy_depth[:, :, 0:2],keypoint_weight[:, :, 0:2]).item())
            depth_error = float(weighted_l1_loss(xy_depth_pred[:, :, 2], keypoint_xy_depth[:, :, 2], keypoint_weight[:, :, 2]).item())
            keypoint_xy_pred_heatmap, _ = predict.heatmap2d_to_normalized_imgcoord_argmax(regress_heatmap)
            xy_error_heatmap = float(weighted_l1_loss(keypoint_xy_pred_heatmap[:, :, 0:2], keypoint_xy_depth[:, :, 0:2], keypoint_weight[:, :, 0:2]).item())
            if idx % 100 == 0:
                print('Iteration %d in epoch %d' % (idx, epoch))
                print('The averaged pixel error is (pixel in 256x256 image): ', 256 * xy_error / len(xy_depth_pred))
                print('The averaged depth error is (mm): ', train_config.depth_image_scale * depth_error / len(xy_depth_pred))
                print('The averaged heatmap argmax pixel error is (pixel in 256x256 image): ', 256 * xy_error_heatmap / len(xy_depth_pred))

            # Update info
            train_error_xy += float(xy_error)
            train_error_depth += float(depth_error)
            train_error_xy_heatmap += float(xy_error_heatmap)

        # The info at epoch level
        print('Epoch %d' % epoch)
        print('The training averaged pixel error is (pixel in 256x256 image): ', 256 * train_error_xy / len(dataset_train))
        print('The training averaged depth error is (mm): ',
              train_config.depth_image_scale * train_error_depth / len(dataset_train))
        print('The training averaged heatmap pixel error is (pixel in 256x256 image): ',
              256 * train_error_xy_heatmap / len(dataset_train))


def main():
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'ckpnt-cap_10k')
    train(checkpoint_dir)

    shutil.rmtree('viz-cap_10k', ignore_errors=True)
    os.makedirs('viz-cap_10k', exist_ok=True)
    visualize(os.path.join(checkpoint_dir, 'checkpoint-116.pth'), 'viz-cap_10k')

    evaluate(os.path.join(checkpoint_dir, 'checkpoint-116.pth'), 'eval-cap_10k.json')



if __name__ == '__main__':
    main()
