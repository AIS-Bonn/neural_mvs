from callbacks.callback import *
from callbacks.vis import *

class VisdomCallback(Callback):

    def __init__(self, experiment_name):
        self.vis=Vis("lnn", 8097)
        # self.iter_nr=0
        self.experiment_name=experiment_name

    def after_forward_pass(self, phase, loss, lr, **kwargs):
        pass 

        # print("loss and smooth loss ", loss, " ", smooth_loss)

       

        # if phase.show_visdom:
            # self.vis.log(phase.iter_nr, loss, "loss_"+phase.name, self.experiment_name, smooth=True, show_every=30)
        # self.vis.log(phase.iter_nr, smooth_loss.item(), "smooth_loss_"+phase.name, self.experiment_name, smooth=True, show_every=30)
        # if phase.grad:
        # self.vis.log(phase.iter_nr, lr, "lr", "lr", smooth=False)

        #show image of the depth 
        # min_val=1.2
        # max_val=1.7
        # self.vis.show_img_from_tensor(output_tensor, "output_tensor", min_val, max_val)
        # self.vis.show_img_from_tensor(gt_tensor, "gt_tensor", min_val, max_val)
        # image_logger = VisdomLogger('image')


    def epoch_ended(self, phase, **kwargs):
        # pass
        # mean_iou=phase.scores.avg_class_iou(print_per_class_iou=False)
        # self.vis.log(phase.epoch_nr, mean_iou, "iou_"+phase.name, "iou_"+phase.name, smooth=False)

        if phase.show_visdom and phase.samples_processed_this_epoch!=0:
            loss_avg_per_eppch = phase.loss_acum_per_epoch/ phase.samples_processed_this_epoch
            self.vis.log(phase.epoch_nr, loss_avg_per_eppch, "loss_"+phase.name, self.experiment_name, smooth=False, show_every=1)
