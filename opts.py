import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('SAMWISE training and inference scripts.', add_help=False)

    # Training hyperparameters
    parser.add_argument('--lr', default=1e-5, type=float,
                        help="Learning rate for optimizer")
    parser.add_argument('--batch_size', default=2, type=int,
                        help="Batch size for training")
    parser.add_argument('--batch_size_val', default=1, type=int,
                        help="Batch size for RIS evaluation")
    parser.add_argument('--num_frames', default=8, type=int,
                        help="Number of frames per training clip")
    parser.add_argument('--weight_decay', default=0, type=float,
                        help="Weight decay (L2 regularization)")
    parser.add_argument('--epochs', default=6, type=int,
                        help="Number of training epochs")
    parser.add_argument('--lr_drop', default=[60000], type=int, nargs='+',
                        help="Epochs at which learning rate should drop")
    parser.add_argument('--clip_max_norm', default=1, type=float,
                        help="Max norm for gradient clipping")


    # Image Encoder: SAM2
    # Model configuration
    parser.add_argument('--sam2_version', default='base', type=str, choices=['tiny', 'base', 'large'],
                        help="Version of SAM2 image encoder to use")
    parser.add_argument('--disable_pred_obj_score', default=False, action='store_true',
                        help="Disable predicted object score")
    parser.add_argument('--motion_prompt', default=False, action='store_true',
                        help="Enable motion-based prompting")

    # Cross Modal Temporal Adapter settings
    parser.add_argument('--HSA', action='store_true', default=False,
                        help="Use Hierarchical Selective Attention (HSA)")
    parser.add_argument('--HSA_patch_size', default=[8, 4, 2], type=int, nargs='+',
                        help="Patch sizes used in HSA")
    parser.add_argument('--adapter_dim', default=256, type=int,
                        help="Dimensionality of adapter layers")
    parser.add_argument('--fusion_stages_txt', default=[4,8,12], type=int,
                        help="Text encoder fusion stages")
    parser.add_argument('--fusion_stages', default=[1,2,3], nargs='+', type=int,
                        help="Fusion stages")


    # Conditional Memory Encoder (CME) settings
    parser.add_argument('--use_cme_head', default=False, action='store_true',
                        help="Use Conditional Memory Encoder (CME)")
    parser.add_argument('--switch_mem', default='reweight', type=str, choices=['all_mask', 'reweight', 'avg'],
                        help="Memory switch strategy")
    parser.add_argument('--cme_decision_window', default=4, type=int,
                        help="Minimum number of frames considered between consecutive CME decisions")


    # dataset settings
    # ['ytvos', 'davis', 'refcoco', 'refcoco+', 'refcocog', 'all']
    # 'all': using the three ref datasets for pretraining

    parser.add_argument('--dataset_file', default='ytvos', type=str,
                        help="Dataset to use: ['ytvos', 'davis', 'refcoco', 'refcoco+', 'refcocog', 'all']")
    parser.add_argument('--coco_path', type=str, default='data/coco',
                        help="Path to COCO dataset")
    parser.add_argument('--ytvos_path', type=str, default='data/ref-youtube-vos',
                        help="Path to YouTube-VOS dataset")
    parser.add_argument('--davis_path', type=str, default='data/ref-davis',
                        help="Path to DAVIS dataset")
    parser.add_argument('--mevis_path', type=str, default='/home/cv2/project/haeun/SAMWISE/data/MeVis_release',
                        help="Path to MeViS dataset")
    parser.add_argument('--max_size', default=1024, type=int,
                        help="Frame size for preprocessing")
    parser.add_argument('--augm_resize', default=False, action='store_true',
                        help="Enable data augmentation with random resizing")

    # General settings
    parser.add_argument('--output_dir', default='output', type=str,
                        help="Directory to save model outputs")
    parser.add_argument('--name_exp', default='default', type=str,
                        help="Experiment name for logging/saving")
    parser.add_argument('--device', default='cuda', type=str,
                        help="Device for computation ('cuda' or 'cpu')")
    parser.add_argument('--seed', default=0, type=int,
                        help="Random seed for reproducibility")
    parser.add_argument('--resume', default='', type=str,
                        help="Path to checkpoint for resuming training or for evaluation")
    parser.add_argument('--resume_optimizer', default=False, action='store_true',
                        help="Resume optimizer state from checkpoint")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help="Epoch to start training from (for resuming)")
    parser.add_argument('--eval', action='store_true',
                        help="Run evaluation instead of training - for RIS only")
    parser.add_argument('--num_workers', default=4, type=int,
                        help="Number of worker threads for data loading")
    parser.add_argument('--no_distributed', action='store_true', default=False,
                        help="Disable distributed training")

    # Testing and evaluation settings
    parser.add_argument('--threshold', default=0.5, type=float,
                        help="Threshold for binary mask predictions")
    parser.add_argument('--split', default='valid', type=str, choices=['valid', 'valid_u', 'test'],
                        help="Dataset split for evaluation")
    parser.add_argument('--visualize', action='store_true',
                        help="Enable mask visualization during inference")
    parser.add_argument('--eval_clip_window', default=8, type=int,
                        help="Frame window size for evaluation")
    parser.add_argument('--set', type=str, default='val',
                        help="Subset to evaluate ('val' or other subsets)")
    parser.add_argument('--task', type=str, default='unsupervised',
                        choices=['semi-supervised', 'unsupervised'],
                        help="Evaluation task type")
    parser.add_argument('--results_path', type=str,
                        help="Path to folder containing the sequences folders results")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed from distributed launcher")

    return parser


