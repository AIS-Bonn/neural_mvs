#!/usr/bin/env python3.6

# import sys
# sys.path.append('../')
# import imageio
# from config import config_parser
# from ibrnet.sample_ray import RaySamplerSingleImage
# from ibrnet.render_image import render_single_image
# from ibrnet.model import IBRNetModel
# from utils import *
# from ibrnet.projection import Projector
# from ibrnet.data_loaders import dataset_dict
# import tensorflow as tf
# from lpips_tensorflow import lpips_tf
from torch.utils.data import DataLoader
from llff_test import LLFFTestDataset
from collections import namedtuple

MyStruct = namedtuple("MyStruct", "llffhold eval_scenes rootdir num_source_views")

# os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':
    # parser = config_parser()
    # args = parser.parse_args()
    # args.distributed = False

    # Create IBRNet model
    # model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = "llff_test"

    # extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    # print("saving results to eval/{}...".format(extra_out_dir))
    # os.makedirs(extra_out_dir, exist_ok=True)

    # projector = Projector(device='cuda:0')

    # assert len(args.eval_scenes) == 1, "only accept single scene"
    # scene_name = "fern"
    # out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step))
    # os.makedirs(out_scene_dir, exist_ok=True)
    args= MyStruct(llffhold=9, eval_scenes="flower", rootdir="/media/rosu/Data/data/nerf", num_source_views=3 )

    test_dataset = LLFFTestDataset(args, 'test', scenes=args.eval_scenes)
    # save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    # results_dict = {scene_name: {}}

    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0]
        # file_id = os.path.basename(rgb_path).split('.')[0]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        gt_rgb = data['rgb'][0]

        print("rgb path is ", rgb_path)