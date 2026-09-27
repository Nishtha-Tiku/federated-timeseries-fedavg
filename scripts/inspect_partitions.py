from collections import Counter
from src.data import load_data
from federated.partition import iid_partition, subject_partition

def show(name,partitions,labels):
    print(f"\n=== {name} ===")
    for cid,indices in partitions.items():
        counts=Counter(labels[indices].tolist()); dist=", ".join(f"class {k}: {v}" for k,v in sorted(counts.items())); print(f"Client {cid}: {len(indices)} samples | {dist}")
def main():
    (_,y,subjects),_=load_data(); labels=y.numpy(); show("IID partition",iid_partition(labels,4,42),labels); show("Subject-based partition",subject_partition(subjects.numpy(),4),labels)
if __name__=="__main__": main()
