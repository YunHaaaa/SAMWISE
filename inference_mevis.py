'''
Inference code for SAMWISE, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
import os
from PIL import Image
import torch.nn.functional as F
import json
from tqdm import tqdm
import sys
from pycocotools import mask as cocomask
from tools.colormap import colormap
import opts
from models.samwise import build_samwise
from util.misc import on_load_checkpoint
from tools.metrics import db_eval_boundary, db_eval_iou
from datasets.transform_utils import VideoEvalDataset
from torch.utils.data import DataLoader
from os.path import join
from datasets.transform_utils import vis_add_mask


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def main(args):
    args.batch_size = 1
    print("Inference only supports for batch size = 1") 
    print(args)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    utils.init_distributed_mode(args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, 'Annotations')
    os.makedirs(save_path_prefix, exist_ok=True)
    args.log_file = join(args.output_dir, 'log.txt')
    with open(args.log_file, 'w') as fp:
        fp.writelines(" ".join(sys.argv)+'\n')
        fp.writelines(str(args.__dict__)+'\n\n')        

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        os.makedirs(save_visualize_path_prefix, exist_ok=True)

    start_time = time.time()
    # model
    model = build_samwise(args)
    device = torch.device(args.device)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if list(checkpoint['model'].keys())[0].startswith('module'):
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}        
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))


    print('Start inference')
    result = eval_mevis(args, model, save_path_prefix, save_visualize_path_prefix, split=split)

    if args.split == 'valid_u':
        J_score, F_score, JF = result[0], result[1], result[2]
        out_str = f'J&F: {JF}\tJ: {J_score}\tF: {F_score}'
        with open(args.log_file, 'a') as fp:
            fp.writelines(out_str+'\n')

    end_time = time.time()
    total_time = end_time - start_time

    print("Total inference time: %.4f s" %(total_time))


def eval_mevis(args, model, save_path_prefix, save_visualize_path_prefix, split='valid_u'):
    # load data
    root = Path(args.mevis_path)
    img_folder = join(root, split, "JPEGImages")
    meta_file = join(root, split, "meta_expressions.json")
    gt_file = join(root, split, 'mask_dict.json')
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    if args.split == 'valid_u':
        with open(gt_file, "r") as f:
            gt_data = json.load(f)        
    else:
        gt_data = None

    video_list = list(data.keys())
    progress = tqdm(
        total=len(video_list),
        ncols=0
    )
    f_log_vid = join(args.output_dir, 'log_metrics_byvid.txt')
    # start inference
    f_log = join(args.output_dir, 'log_metrics.txt')
    model.eval()
    out_dict = {}
    # 1. For each video
    for i_v, video in enumerate(video_list):
        metas = [] # list[dict], length is number of expressions
        expressions = data[video]["expressions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)
        out_dict_per_vid = {}
        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]

            video_len = len(frames)
            all_pred_masks, all_decisions = [], []

            vd = VideoEvalDataset(join(img_folder, video_name), frames, max_size=args.max_size)
            print(img_folder, video_name, frames)
            dl = DataLoader(vd, batch_size=args.eval_clip_window,
                    num_workers=args.num_workers, shuffle=False)
            origin_w, origin_h = vd.origin_w, vd.origin_h
            # 3. for each clip
            for imgs, clip_frames_ids in dl:
                clip_frames_ids = clip_frames_ids.tolist()
                imgs = imgs.to(args.device)  # [eval_clip_window, 3, h, w]
                img_h, img_w = imgs.shape[-2:]
                size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
                target = {"size": size, 'frame_ids': clip_frames_ids}

                with torch.no_grad():
                    outputs = model([imgs], [exp], [target], [img_folder, video_name, frames, exp_id])

                pred_masks = outputs["pred_masks"]  # [t, q, h, w]
                pred_masks = pred_masks.unsqueeze(0)
                pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
                pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu() 
                all_pred_masks.append(pred_masks)

            # store the video results
            all_pred_masks = torch.cat(all_pred_masks, dim=0).numpy()  # (video_len, h, w)
                        
            # load GTs
            if args.split == 'valid_u':
                h, w = all_pred_masks.shape[-2:]
                gt_masks = np.zeros((video_len, h, w), dtype=np.uint8)
                anno_ids = data[video]['expressions'][exp_id]['anno_id']
                for frame_idx, frame_name in enumerate(data[video]['frames']):
                    for anno_id in anno_ids:
                        mask_rle = gt_data[str(anno_id)][frame_idx]
                        if mask_rle:
                            gt_masks[frame_idx] += cocomask.decode(mask_rle)

                j = db_eval_iou(gt_masks, all_pred_masks).mean()
                f = db_eval_boundary(gt_masks, all_pred_masks).mean()
                # print(f'J {j} & F {f}')
                out_dict[exp] = [j, f]
                out_dict_per_vid[exp] = [j, f]
            else:
                # save binary image
                save_path = join(save_path_prefix, video_name, exp_id)
                os.makedirs(save_path, exist_ok=True)
                for j in range(video_len):
                    frame_name = frames[j]
                    mask = all_pred_masks[j].astype(np.float32) 
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frame_name + ".png")
                    mask.save(save_file)
                
            if args.visualize:
                for t, frame in enumerate(frames):
                    # original
                    img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                    source_img = Image.open(img_path).convert('RGBA') # PIL image

                    # draw mask
                    source_img = vis_add_mask(source_img, all_pred_masks[t], color_list[i%len(color_list)])

                    # save
                    save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video, str(i))
                    os.makedirs(save_visualize_path_dir, exist_ok=True)
                    save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                    source_img.save(save_visualize_path)
                    
        if args.split == 'valid_u':
            J_score, F_score, JF = get_current_metrics(out_dict)
            J_score_vid, F_score_vid, JF_vid = get_current_metrics(out_dict_per_vid)
            out_str = f'{i_v}/{len(video_list)} J&F: {JF}\tJ: {J_score}\tF: {F_score}'
            out_str_vid = f'{i_v}/{len(video_list)} CURRENT J&F: {JF_vid}  J: {J_score_vid}  F: {F_score_vid}'
            if utils.is_main_process():
                with open(f_log, 'a') as fp:
                    fp.write(out_str + '\n')
                    fp.write(out_str_vid + '\n')
                with open(f_log_vid, 'a') as fp:
                    fp.writelines(out_str_vid + '\n')
            print('\n' + out_str + '\n' + out_str_vid)
        progress.update(1)

    if args.split == 'valid_u':
        J_score, F_score, JF = get_current_metrics(out_dict)
        print(f'J: {J_score}')
        print(f'F: {F_score}')
        print(f'J&F: {JF}')
        
        return [J_score, F_score, JF]
    return [0, 0, 0]


def get_current_metrics(out_dict):
    j = [out_dict[x][0] for x in out_dict]
    f = [out_dict[x][1] for x in out_dict]

    J_score = np.mean(j)
    F_score = np.mean(f)
    JF = (np.mean(j) + np.mean(f)) / 2
    return J_score, F_score, JF


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)

    main(args)
