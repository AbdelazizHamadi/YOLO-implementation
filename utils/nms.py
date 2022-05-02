import torch
from utils_me.iou import intersection_over_union


def non_max_suppression(bboxes, prob_threshold, box_format="corners", iou_threshold=0.5):
    # bboxes =[[ class=1, confidence score = 0.0 -> 1.0, x1, y1, x2, y2 ], [.., ..., ..]]

    assert type(bboxes) == list, 'only list accepted'

    # remove bounding boxes less than a threshold 
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    # sort by probability
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box
                  for box in bboxes
                  if box[0] != chosen_box[0]
                  or intersection_over_union(torch.tensor(box[2:]),
                                             torch.tensor(chosen_box[2:]),
                                             box_format=box_format) < iou_threshold
                  ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms
