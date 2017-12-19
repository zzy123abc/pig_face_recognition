import numpy as np
import pandas as pd

out=np.load('out.npy')

imgs_test=[]
f=open('imgs_list.txt')
for line in f.readlines():
    line=line.strip('\n')
    imgs_test.append(line)
f.close()

classes = list(range(1, 31))
classes.sort(key = lambda x:str(x))

out=out.transpose()
res=pd.DataFrame(out,index=classes,columns=imgs_test)
res=res.unstack()
res.to_csv('submit.csv')

print 'submit ok'