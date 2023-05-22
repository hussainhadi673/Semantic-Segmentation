# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.confusion_matrix import ConfusionMatrixTensorflow
from src.prepare_data import prepare_data


if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Evaluation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str,
                        required=True,
                        help='Path to the checkpoint of the trained model.')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    _, data_loader, *add_data_loader = prepare_data(args, with_input_orig=True)
    if args.valid_full_res:
        # cityscapes only -> use dataloader that returns full resolution images
        data_loader = add_data_loader[0]

    n_classes = data_loader.dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, args.no_of_class)
    checkpoint = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.to(device)

    n_samples = 0

    confusion_matrices = dict()

    cameras = data_loader.dataset.cameras

    for camera in cameras:
        confusion_matrices[camera] = dict()
        confusion_matrices[camera] = ConfusionMatrixTensorflow(n_classes)
        # confusion_matrices[camera] = ConfusionMatrixTensorflow(40)
        n_samples_total = len(data_loader.dataset)
        with data_loader.dataset.filter_camera(camera):
            for i, sample in enumerate(data_loader):
                n_samples += sample['image'].shape[0]
                print(f'\r{n_samples}/{n_samples_total}', end='')

                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label_orig = sample['label']

                _, image_h, image_w = label_orig.shape

                with torch.no_grad():
                    if args.modality == 'rgbd':
                        inputs = (image, depth)
                    elif args.modality == 'rgb':
                        inputs = (image,)
                    elif args.modality == 'depth':
                        inputs = (depth,)

                    pred = model(*inputs)

                    pred = F.interpolate(pred, (image_h, image_w),
                                         mode='bilinear',
                                         align_corners=False)
                    pred = torch.argmax(pred, dim=1)

                    # ignore void pixels
                    mask = label_orig > 0
                    label = torch.masked_select(label_orig, mask)
                    pred = torch.masked_select(pred, mask.to(device))

                    # In the label 0 is void but in the prediction 0 is wall.
                    # In order for the label and prediction indices to match
                    # we need to subtract 1 of the label.
                    label -= 1

                    # copy the prediction to cpu as tensorflow's confusion
                    # matrix is faster on cpu
                    pred = pred.cpu()

                    label = label.numpy()
                    
                    pred = pred.numpy()

#conversion from 40 to 37 classes
                    for i in range(len(pred)):
                        if int(pred[i]) == 37:
                            pred[i] = 11
                            
                        elif int(pred[i]) == 38:
                            pred[i] = 4
                            
                        elif int(pred[i]) == 39:
                            pred[i] = 5

#conversion from 40 to 13 classes                    
                    # for i in range(len(pred)):
                    #     if int(pred[i]) == 0:
                    #         pred[i] = 11
                            
                    #     elif int(pred[i]) == 1:
                    #         pred[i] = 4
                            
                    #     elif int(pred[i]) == 2:
                    #         pred[i] = 5
                        
                    #     elif int(pred[i]) == 3:
                    #         pred[i] = 0
                            
                    #     elif int(pred[i]) == 4:
                    #         pred[i] = 3
                            
                    #     elif int(pred[i]) == 5:
                    #         pred[i] = 8
                        
                    #     elif int(pred[i]) == 6:
                    #         pred[i] = 9
                            
                    #     elif int(pred[i]) == 7:
                    #         pred[i] = 11
                            
                    #     elif int(pred[i]) == 8:
                    #         pred[i] = 12
                        
                    #     elif int(pred[i]) == 9:
                    #         pred[i] = 5
                            
                        
                    #     elif int(pred[i]) == 10:
                    #         pred[i] = 7
                            
                    #     elif int(pred[i]) == 11:
                    #         pred[i] = 5
                        
                    #     elif int(pred[i]) == 12:
                    #         pred[i] = 12
                            
                    #     elif int(pred[i]) == 13:
                    #         pred[i] = 9
                            
                    #     elif int(pred[i]) == 14:
                    #         pred[i] = 5
                        
                    #     elif int(pred[i]) == 15:
                    #         pred[i] = 12
                            
                    #     elif int(pred[i]) == 16:
                    #         pred[i] = 5
                            
                    #     elif int(pred[i]) == 17:
                    #         pred[i] = 6
                        
                    #     elif int(pred[i]) == 18:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 19:
                    #         pred[i] = 4
                        
                    #     elif int(pred[i]) == 20:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 21:
                    #         pred[i] = 2
                        
                    #     elif int(pred[i]) == 22:
                    #         pred[i] = 1
                            
                    #     elif int(pred[i]) == 23:
                    #         pred[i] = 5
                        
                    #     elif int(pred[i]) == 24:
                    #         pred[i] = 10
                            
                    #     elif int(pred[i]) == 25:
                    #         pred[i] = 6
                        
                    #     elif int(pred[i]) == 26:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 27:
                    #         pred[i] = 6
                        
                    #     elif int(pred[i]) == 28:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 29:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 30:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 31:
                    #         pred[i] = 5
                        
                    #     elif int(pred[i]) == 32:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 33:
                    #         pred[i] = 6
                        
                    #     elif int(pred[i]) == 34:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 35:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 36:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 37:
                    #         pred[i] = 6
                            
                    #     elif int(pred[i]) == 38:
                    #         pred[i] = 5
                            
                    #     elif int(pred[i]) == 39:
                    #         pred[i] = 6
                    
                    confusion_matrices[camera].update_conf_matrix(label, pred)
                print(f'\r{i + 1}/{len(data_loader)}', end='')
            
        miou, ious = confusion_matrices[camera].compute_miou()
        
        iou_common_class = []
        for i in range(len(ious)):
            # print(i)
            if ious[i] != 0:
                iou_common_class.append(ious[i])
        
        
        #print(f'\rCamera: {camera} mIoU: {100*miou:0.2f}')
        
        print(" ")
        print(f'IoU: ', ious)
        #print(f'Common_IoU: ', iou_common_class)
        
        # miou_common_class = np.mean(iou_common_class)
        
        sum_i = 0.00
        for i in iou_common_class:
            sum_i = sum_i + float(i)
        miou_common_class = sum_i/len(iou_common_class)
        
        print(" ")
        print(f'\r Common Class mIoU: {100*miou_common_class:0.2f}')
                

    # confusion_matrices['all'] = ConfusionMatrixTensorflow(n_classes)

    # # sum confusion matrices of all cameras
    # for camera in cameras:
    #     confusion_matrices['all'].overall_confusion_matrix += \
    #         confusion_matrices[camera].overall_confusion_matrix
    # miou, ious = confusion_matrices['all'].compute_miou()
    # print(f'All Cameras, mIoU: {100*miou:0.2f}')
    # print("ious", ious)