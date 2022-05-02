import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):

    def __init__(self, split=7, boxes=2, classes=20):
        super(YoloLoss, self).__init__()

        self.mse = nn.MSELoss(reduction="sum")
        self.split = split
        self.boxes = boxes
        self.classes = classes

        self.lambda_noObj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        # reshape prediction so it be (n, 7, 7, 30). // check the paper work's last layer which means
        # 30 predicted variables for each grid cell
        # 20 variables for class probability [person, dog, ... ect]
        # + 1 (* 2) variables for objectiveness (0 or 1 if there's an object or not in the cell)
        # + 4 variables for each bbox [x, y, w, h, p] . all the previous vars = 30
        # and we have 7 * 7 grid cells ==> (n, 7, 7, 30)
        # 30 variables (tensor) : [person,car,...,19,Prob_box1,x,y,w,h,Prob_box2, x,y,w,h]
        # predictions = predictions.reshape(-1, self.split, self.split, self.classes + self.boxes * 5)

        # pass only the last four elements (box predictions x, y, w, h)
        # get box_one & box_two predictions
        box_one_preds = predictions[..., 21:25]
        box_two_preds = predictions[..., 26:30]

        # ground truth box in that cell
        gt_box_cell = targets[:, :, :, 21:25]

        # probability of a box containing an object
        prob_box_one = predictions[..., 20:21]
        prob_box_two = predictions[..., 25:26]

        # we check if there's an abject in that cell or not (pre-defined in dataset)
        exist_box = targets[..., 20:21]  # I_obj_i (Identity object I)

        # classes
        pred_class = predictions[..., :20]
        gt_class = targets[..., :20]

        # calculate iou between predicted boxes (2) and gt box
        iou_bbox_one = intersection_over_union(box_one_preds, gt_box_cell)
        iou_bbox_two = intersection_over_union(box_two_preds, gt_box_cell)

        ious = torch.cat([iou_bbox_one.unsqueeze(0), iou_bbox_two.unsqueeze(0)], dim=0)

        iou_max, best_box_idx = torch.max(ious, dim=0)

        # best_box_idx output can be 0 or 1:
        #   0 : first box is the best
        #   1 : second box is the best

        #########################
        # FOR OBJECT CORDS LOSS #
        #########################

        # box_prediction output : tensor ([x, y, w, h])
        box_prediction = exist_box * (best_box_idx * box_two_preds +
                                      (1 - best_box_idx) * box_one_preds)

        box_target = exist_box * gt_box_cell

        # square root of width & height
        box_prediction[..., 2:4] = torch.sign(box_prediction[..., 2:4]) * torch.sqrt(
            torch.abs(box_prediction[..., 2:4]) + 1e-6)

        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        # box_loss = self.mse(torch.flatten(box_prediction, end_dim=-2), torch.flatten(box_target, end_dim=-2))

        box_loss = self.mse(box_prediction, box_target)

        #########################
        # FOR OBJECT LOSS       #
        #########################

        pred_box = (
                best_box_idx * prob_box_two + (1 - best_box_idx) * prob_box_one
        )

        object_loss = self.mse(exist_box * pred_box, exist_box * exist_box * iou_max)

        #########################
        # FOR NO OBJECT LOSS    #
        #########################

        no_object_loss = self.mse(
            (1 - exist_box) * prob_box_one,
            (1 - exist_box) * exist_box,
        )

        no_object_loss += self.mse(
            (1 - exist_box) * prob_box_two,
            (1 - exist_box) * exist_box
        )

        #########################
        # FOR OBJECT Class LOSS #
        #########################

        class_loss = self.mse(
            exist_box * pred_class,
            exist_box * gt_class
        )

        loss = (
                self.lambda_coord * box_loss
                + object_loss
                + self.lambda_noObj * no_object_loss
                + class_loss
        )

        return loss
