from datasets import load_dataset
ds = load_dataset('win-wang/Machine_Learning_QA_Collection')
lengths = [len(s) for s in ds['train']['text']]  

lengths.sort()
total = len(lengths)
print(f"median: {lengths[total//2]}")
print(f"90th pct: {lengths[int(total*0.9)]}")
print(f"95th pct: {lengths[int(total*0.95)]}")
print(f"99th pct: {lengths[int(total*0.99)]}")
print(f"max: {lengths[-1]}")