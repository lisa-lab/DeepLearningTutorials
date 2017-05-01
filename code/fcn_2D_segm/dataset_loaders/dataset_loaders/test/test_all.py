import time
import pytest

from dataset_loaders import *
datasets = [el for el in dir() if 'Dataset' in el]


@pytest.mark.parametrize('dataset', [
        eval(dataset) for dataset in datasets])
def testOneDataset(dataset, verbose=False):
    '''Test the data of all the datasets

    This tests all the datasets that are imported through the dataset
    loaders __init__. The test only checks that all the files are
    readable. Run as `pytest -s test_all`'''
    for which_set in ['train', 'val', 'test']:
        print('\n******************************************')
        print('Testing ' + dataset.name + ' - ' + which_set)
        d = dataset(
            which_set=which_set,
            batch_size=1,
            seq_length=0,  # 4D
            raise_IOErrors=True)
        start = time.time()
        tot = 0

        for mb in range(d.nbatches):
            batch = d.next()
            assert batch is not None, 'The batch was empty.'

            # time.sleep approximates running some model
            time.sleep(0.1)
            if verbose:
                stop = time.time()
                part = stop - start - 1
                start = stop
                tot += part
                print('Minibatch %s time: %s (%s)' % (
                    str(mb), part, tot))
