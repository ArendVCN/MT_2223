import os
import torch

'''
Since ESM representations are apparently padded on both sides (sequence of 900 AA became tensor of 902x1280),
the padding has to be removed again.

Script is called with no arguments and will strip the padding and save the new representations into new directories
with suffix `unpadded`. Afterwards the suffix can be removed and the directories will replace the original, padded ones.
'''

path = "/data/home/arendvc/esm_outputs/"
dirs = ["Ram22_" + ds + "_windowed/" for ds in ["train", "test", "valid"]]

for d in dirs:
    for file in os.listdir(path + d):
        if file.endswith(".pt"):
            f = open(path + d + file, "rb")
            rep = torch.load(f, map_location=torch.device('cpu'))
            rep = torch.squeeze(rep)
            rep = rep[1:-1, :]

            new_dir = d.rstrip('/') + "_unpadded/"
            if not os.path.exists(path + new_dir):
                os.mkdir(path + new_dir)
            torch.save(rep, path + new_dir + file)
