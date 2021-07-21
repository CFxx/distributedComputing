
from mpi4py import MPI

from source.functions import load_coco_data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

train_dir = '/Volumes/CF_Lacie_P2/train/'
meta_filename = 'metadata.json'

if rank == 0:
    data = load_coco_data(train_dir, meta_filename, 'id', ['image_id'])[0:98]
    data = [data[0:32], data[33:65], data[66:98]]
else:
    data = None

data = comm.scatter(data, root=0)

print(f'ranke {rank}:')
print(data)


