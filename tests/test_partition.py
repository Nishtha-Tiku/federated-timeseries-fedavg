import numpy as np
from federated.partition import iid_partition, subject_partition

def test_iid_partition_covers_all_samples():
    labels=np.array([0,1,2,0,1,2,0,1]); parts=iid_partition(labels,4); combined=sorted(i for ids in parts.values() for i in ids); assert combined==list(range(len(labels)))
def test_subject_partition_covers_all_samples():
    subjects=np.array([1,1,2,2,3,3,4,4]); parts=subject_partition(subjects,2); combined=sorted(i for ids in parts.values() for i in ids); assert combined==list(range(len(subjects)))
