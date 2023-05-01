
"""

reference: https://github.com/epfml/ChocoSGD/blob/0557423bded53687c8955fcf46487779cc29ff07/dl_code/pcode/utils/sparsification.py#L86
"""

import math
import numpy as np
import torch

def get_n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()


"""define some general compressors, e.g., top_k, random_k, sign"""


class SparsificationCompressor(object):
    def get_top_k(self, x, ratio):
        """it will sample the top 1-ratio of the samples."""
        x_data = x.view(-1)
        x_len = x_data.nelement()
        top_k = max(1, int(x_len * (1 - ratio)))

        # get indices and the corresponding values
        if top_k == 1:
            _, selected_indices = torch.max(x_data.abs(), dim=0, keepdim=True)
        else:
            _, selected_indices = torch.topk(
                x_data.abs(), top_k, largest=True, sorted=False
            )
        #print(x.size(), top_k)
        return x_data[selected_indices], selected_indices

    def get_mask(self, flatten_arr, indices):
        mask = torch.zeros_like(flatten_arr)
        mask[indices] = 1

        mask = mask.byte()
        return mask.float(), (~mask).float()

    def get_random_k(self, x, ratio, is_biased=True):
        """it will randomly sample the 1-ratio of the samples."""
        # get tensor size.
        x_data = x.view(-1)
        x_len = x_data.nelement()
        top_k = max(1, int(x_len * (1 - ratio)))

        # random sample the k indices.
        selected_indices = np.random.choice(x_len, top_k, replace=False)
        selected_indices = torch.LongTensor(selected_indices).to(x.device)

        if is_biased:
            return x_data[selected_indices], selected_indices
        else:
            return x_len / top_k * x_data[selected_indices], selected_indices

    def compress(self, arr, op, compress_ratio, is_biased):
        if "top_k" in op:
            values, indices = self.get_top_k(arr, compress_ratio)
        elif "random_k" in op:
            values, indices = self.get_random_k(arr, compress_ratio)
        else:
            raise NotImplementedError

        # n_bits = get_n_bits(values) + get_n_bits(indices)
        return values, indices


class QuantizationCompressor(object):
    def get_qsgd(self, x, s, is_biased=False):
        # s=255 for level=8
        norm = x.norm(p=2)
        level_float = s * x.abs() / norm
        previous_level = torch.floor(level_float)
        is_next_level = (torch.rand_like(x) < (level_float - previous_level)).float()
        new_level = previous_level + is_next_level
        # assert not torch.isnan(is_next_level).any()
        #print('\n',x, new_level/s)

        scale = 1
        if is_biased:
            d = x.nelement()
            scale = 1.0 / (min(d / (s ** 2), math.sqrt(d) / s) + 1.0)
            #print(scale)

        return scale * torch.sign(x) * norm * (new_level / s)


    def compress(self, arr, op, quantize_level, is_biased):
        if quantize_level != 32:
            s = 2 ** quantize_level - 1
            values = self.get_qsgd(arr, s, is_biased)
        else:
            values = arr
        return values

    def uncompress(self, arr):
        return arr



   
