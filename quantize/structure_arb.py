import torch
from quantize.autosearch import structural_searching_multip_alternating_group_x

import logging
logger = logging.getLogger()

'''
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
'''
def structural_guassian_distribution_multip_alternating_group_x(tmp, H=None, metric="hessian", up_lim=50, num_p=1, inp=None, order2_group=False):
    if (metric == "hessian") :
        target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    elif metric == "magnitude":
        target_weights = tmp
    else:
        raise NotImplementedError

    # print(f'debug', inp)
    optimal_split_list, mask_list = structural_searching_multip_alternating_group_x(target_weights, up_lim, num_p, inp, order2_group=order2_group)

    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    return mask_list
