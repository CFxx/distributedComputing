
from mpi4py import MPI

from source.functions import load_coco_data
from source.functions import create_data_loader
from source.classes.herbarium import Herbarium

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

train_dir = '/Volumes/CF_Lacie_P2/train/'
test_dir = '/Volumes/CF_Lacie_P2/test/'
meta_filename = 'metadata.json'

if rank == 0:
    coco_data, nb_classes = load_coco_data(train_dir, meta_filename)
    coco_data = coco_data[0:98]
    #nb_classes = len(coco_data['category_id'].value_counts())
    coco_data = [coco_data[0:32], coco_data[33:65], coco_data[66:98]]
else:
    coco_data = None
    nb_classes = 0
coco_data = comm.scatter(coco_data, root=0)
nb_classes = comm.bcast(nb_classes, root=0)

print(f'rank {rank} received {nb_classes} of classes :')
#print(coco_data)

print(f'rank {rank} model initialization.')
herb = Herbarium(train_dir, test_dir, meta_filename)
print(f'rank {rank} prepare loader.')
herb.train_data_getter, _= create_data_loader(train_dir, coco_data, 'file_name', 'category_id', herb.transform, herb.batch)
herb.nb_classes = nb_classes
herb.init_model()
print(f'rank {rank} step train.')
herb.step_train()
print(f'rank {rank} extracting weights.')
weights = herb.model.state_dict()
#print('weight before bcast :', len(weights))
all_weights = comm.allgather(weights)

total_instance = len(all_weights)
print(total_instance)
nb_weights = len(all_weights[0])

for key in weights:
    sum = 0
    for other_weights in all_weights:
        sum += other_weights[key]
    weights[key] = sum/total_instance

print(f'new averaged weights have been updated')
herb.model.load_state_dict(weights)

herb.step_train()

input()

