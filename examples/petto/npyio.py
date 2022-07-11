import numpy as np

def write_df(vec):

    rank=vec[0]
    np.save("df_%d.npy"%rank,vec[1:])
    return 0

def write_dg(vec):

    rank=vec[0]
    np.save("dg_%d.npy"%rank,vec[1:])
    return 0

def write_fg(vec):
    np.save("fg.npy",vec)
    return 0

def read(vec):
    rank=vec[-1]
    ranges=vec[:-1]
    tmp2=np.zeros((ranges[rank+1]-ranges[rank]),dtype=float)
    tmp=np.load("xxx.npy")
    tmp2[:]=tmp[ranges[rank]:ranges[rank+1]]
#   print(len(tmp2), len(tmp[ranges[rank]:ranges[rank+1]]))
    return tmp2
#   return np.load("xxx.npy")

    
