# -*- coding: utf-8 -*-
import math
import functools

import numpy as np
import torch
import torch.distributed as dist


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(
        self,
        seed,
        data,
        partition_sizes,
        non_iid_alpha = 1.0,
        partition_type="non_iid_dirichlet",
        
    ):
        # prepare info.
        self.random_state       = np.random.RandomState(seed)
        self.data               = data
        self.non_iid_alpha      = non_iid_alpha
        self.partition_sizes    = partition_sizes
        self.partition_type     = partition_type
        self.partitions         = []

        # get data, data_size, indices of the data.
        self.data_size = len(data)
        if type(data) is not Partition:
            self.data = data
            indices   = np.array([x for x in range(0, self.data_size)])
        else:
            self.data = data.data
            indices   = data.indices
        # apply partition function.
        self.partition_indices(indices)

    def partition_indices(self, indices):
        indices     = self._create_indices(indices)

        # partition indices.
        from_index     = 0
        for partition_size in self.partition_sizes:
            to_index   = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index

        # display the class distribution over the partitions.
        self.targets_of_partitions = record_class_distribution(
                self.partitions,
                self.data.targets if hasattr(self.data, "targets") else self.data.golds,
            )

    def _create_indices(self, indices):
        if self.partition_type   == "origin":
            pass
        elif self.partition_type == "random":
            # it will randomly shuffle the indices.
            self.random_state.shuffle(indices)
        elif self.partition_type == "sorted":
            # it will sort the indices based on the data label.
            indices = [
                i[0]
                for i in sorted(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ],
                    key=lambda x: x[1],
                )
            ]
        elif self.partition_type == "non_iid_dirichlet":
            num_indices = len(indices)
            n_workers = len(self.partition_sizes)

            targets = (
                self.data.targets if hasattr(self.data, "targets") else self.data.golds
            )
            num_classes = len(np.unique(targets))
            indices2targets = np.array(list(enumerate(targets)))

            list_of_indices = build_non_iid_by_dirichlet(
                random_state=self.random_state,
                indices2targets=indices2targets,
                non_iid_alpha=self.non_iid_alpha,
                num_classes=num_classes,
                num_indices=num_indices,
                n_workers=n_workers,
            )
            indices = functools.reduce(lambda a, b: a + b, list_of_indices)
        else:
            raise NotImplementedError(
                f"The partition scheme={self.partition_type} is not implemented yet"
            )
        return indices

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind]), self.targets_of_partitions[partition_ind]


def build_non_iid_by_dirichlet(
    random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    n_auxi_workers = 10

    # random shuffle targets indices.
    random_state.shuffle(indices2targets)

    # partition indices.
    from_index       = 0
    splitted_targets = []
    num_splits       = math.ceil(n_workers / n_auxi_workers)
    split_n_workers  = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets      = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers  = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def record_class_distribution(partitions, targets):
    targets_of_partitions = {}
    targets_np = np.array(targets)
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(
            targets_np[partition], return_counts=True
        )
        targets_of_partitions[idx] = list(zip(unique_elements, counts_elements))
    # print(
    #     f"the histogram of the targets in the partitions: {targets_of_partitions.items()}"
    # )
    return targets_of_partitions


# class Partition(object):
#     def __init__(self, data, index):
#         self.data = data
#         self.index = index
    
#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, index):
#         data_idx = self.index[index]
#         return self.data[data_idx]

# def skew_sort(indices, skew, classes, class_size, seed):
#     # skew belongs to [0,1]
#     rng = random.Random()
#     rng.seed(seed)
#     class_indices = {}
#     for i in range(0, classes):
#         class_indices[i]=indices[0:class_size[i]]
#         indices = indices[class_size[i]:]
#     random_indices = []
#     sorted_indices = []
#     for i in range(0, classes):
#         sorted_size    = int(skew*class_size[i])
#         sorted_indices = sorted_indices + class_indices[i][0:sorted_size]
#         random_indices = random_indices + class_indices[i][sorted_size:]
#     rng.shuffle(random_indices)
#     return random_indices, sorted_indices
            
    
# class DataPartitioner(object):
#     """ Partitions a dataset into different chunks"""
#     def __init__(self, data, sizes, skew, classes, class_size, seed, device):
        
#         self.data = data
#         self.partitions = []
#         data_len = len(data)
#         dataset = torch.utils.data.DataLoader(data, batch_size=512, shuffle=False, num_workers=2)
#         labels = []
#         for batch_idx, (inputs, targets) in enumerate(dataset):
#               labels = labels+targets.tolist()
#         #labels  = [data[i][1] for i in range(0, data_len)]
#         sort_index = np.argsort(np.array(labels))
#         indices = sort_index.tolist()
#         indices_rand, indices = skew_sort(indices, skew=skew, classes=classes, class_size=class_size, seed=seed)
        
#         for i, frac in enumerate(sizes):
#             if skew==1:
#                 part_len = int(frac*data_len)
#                 self.partitions.append(indices[0:part_len])
#                 if len(sizes)>10 and i<10:
#                     #print('here', i, len(sizes), len(indices))
#                     indices = indices[2*part_len:]+indices[part_len:2*part_len]
#                 else:
#                     indices = indices[part_len:]
#             elif skew==0:
#                 part_len = int(frac*data_len)
#                 self.partitions.append(indices_rand[0:part_len])
#                 indices_rand = indices_rand[part_len:] 
#             else:
#                 part_len = int(frac*data_len*skew); 
#                 part_len_rand = int(frac*data_len*(1-skew))
#                 part_ind = indices[0:part_len]+indices_rand[0:part_len_rand]
#                 self.partitions.append(part_ind)
#                 indices = indices[part_len:]
#                 indices_rand = indices_rand[part_len_rand:]

#     def use(self, partition):
#         return Partition(self.data, self.partitions[partition])