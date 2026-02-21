import torch
import h5py
from torch_geometric.data import Data
from torch_geometric.data import Dataset


def create_radius_graph(event_row, radius=0.4):
    particles = event_row[:2100].reshape(-1, 3)
    particles = particles[particles[:, 0] > 0]

    if len(particles) < 2:
        return None

    x = torch.tensor(particles, dtype=torch.float)
    eta_phi = x[:, 1:3]
    dist_matrix = torch.cdist(eta_phi, eta_phi, p=2.0)
    edge_index = torch.nonzero(dist_matrix <= radius, as_tuple=False).t().contiguous()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return Data(x=x, edge_index=edge_index)

class RawChunkDataset(Dataset):
    """
    Blueprint for loading 110+ graph chunks from the LHCO 2020 dataset.
    Uses a least-recently-used (LRU) cache to manage memory on local machines.
    """
    def __init__(self, chunk_dir, indices, max_cache=20):
        super().__init__(None, None, None)
        self.chunk_dir = chunk_dir
        self.data_indices = indices  
        self.chunk_size = 10000 
        self.cache = {}
        self.max_cache = max_cache
        
    def len(self):
        return len(self.data_indices)  
        
    def get(self, idx):
        real_idx = self.data_indices[idx]  
        chunk_idx = real_idx // self.chunk_size
        in_chunk_idx = real_idx % self.chunk_size
        
        # Memory management: Load chunk only if not in cache
        if chunk_idx not in self.cache:
            if len(self.cache) >= self.max_cache:
                # Evict oldest chunk to prevent RAM overflow
                self.cache.pop(next(iter(self.cache))) 
            
            chunk_path = f"{self.chunk_dir}/data_chunk_{chunk_idx}.pt"
            # weights_only=False is required for complex Graph objects
            self.cache[chunk_idx] = torch.load(chunk_path, map_location='cpu', weights_only=False)
            
        return self.cache[chunk_idx][in_chunk_idx]


class LHCH5Dataset(Dataset):
    def __init__(self, file_path, start_idx=0, num_events=100000, radius=0.4):
        super().__init__(None, None, None)
        self.file_path = file_path
        self.start_idx = start_idx
        self.num_events = num_events
        self.radius = radius

    def len(self):
        return self.num_events

    def get(self, idx):
        with h5py.File(self.file_path, 'r') as h5_file:
            dset = h5_file['df']['block0_values']
            row = dset[self.start_idx + idx, :]

        label = torch.tensor([row[2100]], dtype=torch.float)
        graph = create_radius_graph(row, radius=self.radius)
        if graph is not None:
            graph.y = label
            return graph

        return Data(
            x=torch.zeros((1, 3), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=label,
        )