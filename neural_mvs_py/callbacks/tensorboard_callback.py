from instant_ngp_2_py.callbacks.callback import *
from torch.utils.tensorboard import SummaryWriter

from neural_mvs.utils import *


class TensorboardCallback(Callback):

    def __init__(self, experiment_name):
        self.tensorboard_writer = SummaryWriter("tensorboard_logs/"+experiment_name)
        self.experiment_name=experiment_name
        

    def after_forward_pass(self, phase, loss, loss_rgb, psnr, lr, rgb_pred, rgb_gt, confidence_map, point3d,  **kwargs):
        # self.vis.log(phase.iter_nr, loss, "loss_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=True,  show_every=10, skip_first=10)
        # self.vis.log(phase.iter_nr, loss_dice, "loss_dice_"+phase.name, "loss_"+phase.name+"_"+self.experiment_name, smooth=True,  show_every=10, skip_first=10)
        # if phase.grad:
            # self.vis.log(phase.iter_nr, lr, "lr", "loss_"+phase.name+"_"+self.experiment_name, smooth=False,  show_every=30)

        # self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/lr', lr, phase.iter_nr)

        self.tensorboard_writer.add_scalar('neural_mvs/' + phase.name + '/loss', loss, phase.iter_nr)
        if loss_rgb!=0:
            self.tensorboard_writer.add_scalar('neural_mvs/' + phase.name + '/loss_rgb', loss_rgb, phase.iter_nr)
        if psnr!=0:
            self.tensorboard_writer.add_scalar('neural_mvs/' + phase.name + '/psnr', psnr, phase.iter_nr)

        #show imgs
        if (phase.iter_nr<=1 or phase.iter_nr%500==0) and rgb_pred is not None:
            self.tensorboard_writer.add_image('neural_mvs/' + phase.name + '/rgb_pred', rgb_pred.squeeze(0), phase.iter_nr)
        if (phase.iter_nr<=1 or phase.iter_nr%500==0) and rgb_gt is not None:
            self.tensorboard_writer.add_image('neural_mvs/' + phase.name + '/rgb_gt', rgb_gt.squeeze(0), phase.iter_nr)
        if (phase.iter_nr<=1 or phase.iter_nr%500==0) and confidence_map is not None:
            self.tensorboard_writer.add_image('neural_mvs/' + phase.name + '/confidence', confidence_map.squeeze(0), phase.iter_nr)
        if (phase.iter_nr<=1 or phase.iter_nr%500==0) and point3d is not None:
            normal_img=compute_normal(point3d)
            normal_vis=(normal_img+1.0)*0.5
            self.tensorboard_writer.add_image('neural_mvs/' + phase.name + '/normal', normal_vis.squeeze(0), phase.iter_nr)
        
        

      


    def epoch_ended(self, phase, **kwargs):
        pass
        # mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        # self.tensorboard_writer.add_scalar('instant_ngp_2/' + phase.name + '/mean_iou', mean_iou, phase.epoch_nr)
        # self.vis.log(phase.epoch_nr, mean_iou, "iou_"+phase.name,  "loss_"+phase.name+"_"+self.experiment_name, smooth=False,  show_every=1)