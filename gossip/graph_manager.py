
"""
Graph Manager Class

:description: Class provides an API for loading different peer-to-peer
    communication topologies, and cycling through peers.
"""

from math import log as mlog
import torch
import torch.distributed as dist

from .utils import is_power_of


class Edge(object):

    def __init__(self, local_master_rank, dest, src, local_rank, devices):
        self.dest = dest
        self.src = src
        self.process_group = dist.new_group([src, dest])
        if local_master_rank in [self.src, self.dest] and local_rank == 0:
            initializer_tensor = torch.Tensor([1]).to(torch.device("cuda:{}".format(local_master_rank%devices)))
            dist.all_reduce(initializer_tensor, group=self.process_group)
            initializer_tensor = torch.Tensor([1]).to(torch.device("cuda:{}".format(local_master_rank%devices))).half()
            dist.all_reduce(initializer_tensor, group=self.process_group)


class GraphManager(object):

    def __init__(self, rank, world_size, devices, nprocs_per_node=1, local_rank=0, peers_per_itr=2):
        assert int(peers_per_itr) >= 1
        self.rank = rank
        self.world_size = world_size
        self.phone_book = [[] for _ in range(self.world_size)]
        self._peers_per_itr = peers_per_itr
        self._group_indices = [i for i in range(peers_per_itr)]
        self.nprocs_per_node = nprocs_per_node
        self.local_rank = local_rank
        self.devices = devices
        self._make_graph()

    @property
    def peers_per_itr(self):
        return self._peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v):
        self._peers_per_itr = v
        # set group-indices attr. --- point to out-peers in phone-book
        self._group_indices = [i for i in range(v)]

    def _make_graph(self):
        """
        Returns a nested list of peers; the outer-list is indexed by rank,
        the inner list denotes the set of peers that 'rank' can send
        messages to at any point in time
        """
        raise NotImplementedError

    def _add_peers(self, rank, peers):
        for peer in peers:
            if peer not in self.phone_book[rank]:
                self.phone_book[rank].append(Edge(
                    local_master_rank=(self.rank * self.nprocs_per_node),
                    dest=(peer * self.nprocs_per_node),
                    src=(rank * self.nprocs_per_node),
                    local_rank=self.local_rank, devices=self.devices))

    def is_regular_graph(self):
        """ Whether each node has the same number of in-peers as out-peers """
        raise NotImplementedError

    def is_bipartite_graph(self):
        """ Whether graph is bipartite or not """
        raise NotImplementedError

    def is_passive(self, rank=None):
        """ Whether 'rank' is a passive node or not """
        raise NotImplementedError

    def is_dynamic_graph(self, graph_type=None):
        """ Whether the graph-type is dynamic (as opposed to static) """
        raise NotImplementedError

    def get_peers(self, rotate=False):
        """ Returns the out and in-peers corresponding to 'self.rank' """
        # cycle through in- and out-peers by updating group-index
        if rotate:
            self._rotate_group_indices()

        # get out- and in-peers using new group-indices
        out_peers, in_peers = [], []
        for group_index in self._group_indices:
            out_peers.append(self.phone_book[self.rank][group_index].dest)
            for rank, peers in enumerate(self.phone_book):
                if rank == self.rank:
                    continue
                if self.rank * self.nprocs_per_node == peers[group_index].dest:
                    in_peers.append(rank)
        return out_peers, in_peers

    def get_edges(self, rotate=False):
        """ Returns the pairwise process groups between rank and the out and
        in-peers corresponding to 'self.rank' """
        # cycle through in- and out-peers by updating group-index
        if rotate:
            self._rotate_group_indices()

        # get out- and in-peers using new group-indices
        out_edges, in_edges = [], []
        for group_index in self._group_indices:
            out_edges.append(
                self.phone_book[self.rank][group_index])
            for rank, edges in enumerate(self.phone_book):
                if rank == self.rank:
                    continue
                if self.rank * self.nprocs_per_node == edges[group_index].dest:
                    in_edges.append(
                        self.phone_book[rank][group_index])
        return out_edges, in_edges

    def _rotate_group_indices(self):
        """ Incerement group indices to point to the next out-peer """
        increment = self.peers_per_itr
        for i, group_index in enumerate(self._group_indices):
            self._group_indices[i] = int((group_index + increment)
                                         % len(self.phone_book[self.rank]))

    def _rotate_forward(self, r, p):
        """ Helper function returns peer that is p hops ahead of r """
        return (r + p) % self.world_size

    def _rotate_backward(self, r, p):
        """ Helper function returns peer that is p hops behind r """
        temp = r
        for _ in range(p):
            temp -= 1
            if temp < 0:
                temp = self.world_size - 1
        return temp


class RingGraph(GraphManager):

    def _make_graph(self):
        for rank in range(self.world_size):
            f_peer = self._rotate_forward(rank, 1)
            b_peer = self._rotate_backward(rank, 1)
            self._add_peers(rank, [f_peer, b_peer])

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return False

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return False
    
    
    
class GridGraph(GraphManager):

    def _make_graph(self):
        for rank in range(self.world_size):
            a = self._rotate_forward(rank, 1)
            b = self._rotate_backward(rank, 1)
            c = self._rotate_forward(rank, 4)
            d = self._rotate_backward(rank, 4)
     
            self._add_peers(rank, [a,b,c,d])

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return False

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return False


class BipartiteGraph(GraphManager):

    def _make_graph(self):
        even = []
        odd  = []
        for rank in range(self.world_size):
            if (rank+1)%2==0:
                odd.append(rank)
            else:
                even.append(rank)

        for rank in range(self.world_size):
            if (rank+1)%2==0:
                self._add_peers(rank, even)
            else:
                self._add_peers(rank, odd)  

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return True

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return False


class FullGraph(GraphManager):

    def _make_graph(self):
        for rank in range(self.world_size):
            peers = []
            for i in range(1,self.world_size):
                peers.append(self._rotate_forward(rank, i))
            self._add_peers(rank, peers)
            
    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return False

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return False


class ChainGraph(GraphManager):

    def _make_graph(self):
        for rank in range(self.world_size):
            if rank==0:
                f_peer = self._rotate_forward(rank, 1)
                self._add_peers(rank, [f_peer, f_peer])
            elif rank == self.world_size-1:
                b_peer = self._rotate_backward(rank, 1)
                self._add_peers(rank, [b_peer, b_peer])
            else: 
                f_peer = self._rotate_forward(rank, 1)
                b_peer = self._rotate_backward(rank, 1)
                self._add_peers(rank, [f_peer, b_peer])

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return False

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return False


class PetersenGraph(GraphManager):

    def _make_graph(self):
        for rank in range(self.world_size):
            if 0<=rank<4:
                f_peer = self._rotate_forward(rank, 1)
                b_peer = self._rotate_backward(rank, 1)
                d_peer = self._rotate_forward(rank, 5)
                self._add_peers(rank, [f_peer, b_peer, d_peer])
            elif 5<=rank<9:
                f1_peer = self._rotate_forward(rank, 2)
                f2_peer = self._rotate_forward(rank, 3)
                t_peer = rank-5
                self._add_peers(rank, [f1_peer, f2_peer, t_peer])
            elif rank==4:
                f_peer = 0
                b_peer = self._rotate_backward(rank, 1)
                d_peer = self._rotate_forward(rank, 5)
                self._add_peers(rank, [f_peer, b_peer, d_peer])
            elif rank==9:
                f1_peer = self._rotate_backward(rank, 2)
                f2_peer = self._rotate_backward(rank, 3)
                t_peer = rank-5
                self._add_peers(rank, [f1_peer, f2_peer, t_peer])
            else:
                raise NotImplementedError

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return False

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return False


class DyckGraph(GraphManager):

    def _make_graph(self):
        #print(self.world_size)
        if self.world_size==32:
           self._add_peers(0, [1, 7, 8])
           self._add_peers(1, [2, 0, 17])
           self._add_peers(2, [3, 1, 10])
           self._add_peers(3, [4, 2, 19])
           self._add_peers(4, [5, 3, 12])
           self._add_peers(5, [6, 4, 21])
           self._add_peers(6, [7, 5, 14])
           self._add_peers(7, [0, 6, 23])

           self._add_peers(8,  [9,  15, 0])
           self._add_peers(9,  [10, 8,  25])
           self._add_peers(10, [11, 9,  2])
           self._add_peers(11, [12, 10, 27])
           self._add_peers(12, [13, 11, 4])
           self._add_peers(13, [14, 12, 29])
           self._add_peers(14, [15, 13, 6])
           self._add_peers(15, [8,  14, 31])

           self._add_peers(16, [17, 23, 24])
           self._add_peers(17, [18, 16, 1])
           self._add_peers(18, [19, 17, 26])
           self._add_peers(19, [20, 18, 3])
           self._add_peers(20, [21, 19, 28])
           self._add_peers(21, [22, 20, 5])
           self._add_peers(22, [23, 21, 30])
           self._add_peers(23, [16, 22, 7])

           self._add_peers(24, [25, 31, 16])
           self._add_peers(25, [26, 24, 9])
           self._add_peers(26, [27, 25, 18])
           self._add_peers(27, [28, 26, 11])
           self._add_peers(28, [29, 27, 20])
           self._add_peers(29, [30, 28, 13])
           self._add_peers(30, [31, 29, 22])
           self._add_peers(31, [24, 30, 15])
            
        else:
            raise NotImplementedError

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return False

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return False