
from mpi4py import MPI

from source.functions import load_coco_data
from source.functions import create_data_loader
from source.classes.herbarium import Herbarium

import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

train_dir = '/Volumes/CF_Lacie_P2/train/'
test_dir = '/Volumes/CF_Lacie_P2/test/'
meta_filename = 'metadata.json'
epoch = 3

if rank == 0:
    coco_data, nb_classes = load_coco_data(train_dir, meta_filename)
    coco_data = coco_data[0:4096]
    coco_data = coco_data.sample(frac=1).reset_index(drop=True)
    #nb_classes = len(coco_data['category_id'].value_counts())
    coco_data = [coco_data[0:1023], coco_data[1024:2047], coco_data[2048:3071], coco_data[3072:4096]]
else:
    coco_data = None
    nb_classes = 0

# distribute the data to each node(root will receive too)
coco_data = comm.scatter(coco_data, root=0)
# send the number of classes to each node as the number of classes relates to the whole dataset and not partial
nb_classes = comm.bcast(nb_classes, root=0)

#print(f'rank {rank} received {nb_classes} of classes :')

#print(f'rank {rank} model initialization.')
herb = Herbarium(train_dir, test_dir, meta_filename)
#print(f'rank {rank} prepare loader.')
herb.train_data_getter, _= create_data_loader(train_dir, coco_data, 'file_name', 'category_id', herb.transform, herb.batch)
herb.nb_classes = nb_classes
herb.init_model()

start = time.time()
# training
print(f'rank {rank}')
for i in range(epoch):
    print(f'Epoch {i}:')
    herb.step_train(verbose=True)
    weights = herb.model.state_dict()
    all_weights = comm.allgather(weights)
    nb_weights = len(all_weights[0])

    for key in weights:
        sum = 0
        for other_weights in all_weights:
            sum += other_weights[key]
        weights[key] = sum/len(all_weights)

    #print(f'new averaged weights have been updated')
    herb.model.load_state_dict(weights)
    comm.barrier()

end = time.time()
print(f'training time {end-start}s')