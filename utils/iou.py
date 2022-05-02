import torch


def intersection_over_union(box_preds, box_labels, box_format="midpoint"):
    
    # boxes_pred shape (N, 4)
    # boxes_label shape (N, 4)
    
    if box_format == "corners":
        
        # box 1
        box1_x1 = box_preds[..., 0:1]
        box1_y1 = box_preds[..., 1:2]
        box1_x2 = box_preds[..., 2:3]
        box1_y2 = box_preds[..., 3:4]
    
        # box 2
        box2_x1 = box_labels[..., 0:1]
        box2_y1 = box_labels[..., 1:2]
        box2_x2 = box_labels[..., 2:3]
        box2_y2 = box_labels[..., 3:4]
        
    elif box_format == "midpoint":
        
        box1_x1 = box_preds[..., 0:1] - box_preds[..., 2:3] / 2
        box1_y1 = box_preds[..., 1:2] - box_preds[..., 3:4] / 2
        box1_x2 = box_preds[..., 0:1] + box_preds[..., 2:3] / 2
        box1_y2 = box_preds[..., 1:2] + box_preds[..., 3:4] / 2
        box2_x1 = box_labels[..., 0:1] - box_labels[..., 2:3] / 2
        box2_y1 = box_labels[..., 1:2] - box_labels[..., 3:4] / 2
        box2_x2 = box_labels[..., 0:1] + box_labels[..., 2:3] / 2
        box2_y2 = box_labels[..., 1:2] + box_labels[..., 3:4] / 2
         
         
    # find max and min of coordiantes 
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.max(box1_x2, box2_x2)
    y2 = torch.max(box1_y2, box2_y2)
    
    # calculate intersection
    
    # clamp(0) if the substraction is less than zero (boxes do not intersect) take it as zero 
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # calculate area 
    box1_area = abs((box1_x1 - box1_x2) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x1 - box2_x2) * (box2_y1 - box2_y2))
    
    iou = intersection/(box1_area + box2_area - intersection + 1e-6)
    
    return iou 
    

## testing iou score (it works)
box_one = torch.tensor([250, 250, 300, 300])
box_two = torch.tensor([200, 200, 400, 400])

iou_score = intersection_over_union(box_one,box_two).numpy()[0]

    
    