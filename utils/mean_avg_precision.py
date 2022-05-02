import torch
from collections import Counter
from utils_me.iou import intersection_over_union


def mean_average_precision(pred_boxes,
                           true_boxes, iou_threshold=0.5,
                           box_format='corners', num_classes=20):
    # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2],...]

    # train_idx  : the idx of image 
    # class_pred : class name -> 0, 1, ..etc

    average_precisions = []
    epsilon = 1e-10

    # sort detection & ground truths by class 
    for c in range(num_classes):

        # preparing list 
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # count the amount of ground truth boxes in each image
        # let's say image 0 has 3 bboxes 
        # and image 1 has 5 bboxes 
        # so we make a dictionary and use image_idx as key 

        amount_gt_boxes = Counter(gt[0] for gt in ground_truths)
        # amount_gt_boxes (example) -> dict = {0:3, 1:5}

        for key, val in amount_gt_boxes.items():
            amount_gt_boxes[key] = torch.zeros(val)
            # amount_gt_boxes = Counter({0: tensor([0., 0., 0.]), 1: tensor([0., 0., 0., 0., 0.])})

            # tensor initialized with zeros (ground truths not yet associated to a detection)

            # a way to know which ground truth belongs to the predicted detection (with iou score) one of the
            # solutions is creating tensor to distinguish between every ground truth box if 0 : ground truth box is
            # not yet associated to detection if 1 : ground truth box is associated to detection (to avoid assigning
            # multiple detection to one ground truth box association : if we find iou between detection and gt box
            # more than a specified threshold @0.5:0.6:..etc if tensor value of that ground truth is 0 : count it as
            # TP 1 (already assigned):  count it as FP (to avoid assigning multiple detection to one ground truth box)

        # sort detection by confidence score     
        detections = sorted(detections, key=lambda x: x[2], reverse=True)

        # prepare metric variables 
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        # get detections one by one 
        for detection_idx, detection in enumerate(detections):

            # keep only the ground truths of the corresponding image (same image of that detection)
            ground_truths_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_gts = len(ground_truths_img)
            best_iou = 0
            best_gt_idx = 0

            # get gt of the corresponding image one by one 
            # to calculate iou between the detection and every ground truth box in the image

            for idx, gt in enumerate(ground_truths_img):

                # calculate iou between the detection & ground truth box
                iou = intersection_over_union(torch.tensor(detection[3:]),
                                              torch.tensor(gt[3:]),
                                              box_format=box_format
                                              )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:

                # update gt tensor with the correponding image index and the 
                if amount_gt_boxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_gt_boxes[detection[0]][best_gt_idx] = 1

                    # example : amount_gt_boxes = Counter({0: tensor([0., 1.0, 0.]), 1: tensor([0., 1.0, 0., 0., 0.])})

                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # calculate area under the curve (y,x)
        # trapz for intergration 
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
