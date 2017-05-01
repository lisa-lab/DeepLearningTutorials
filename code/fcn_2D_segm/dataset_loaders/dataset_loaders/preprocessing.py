import os, sys
import os.path

mainpath = '/data/lisatmp3/negar/Datasets/moseg/'
exprset = 'Testset'
dspath=mainpath+exprset
target = open('/data/lisatmp3/negar/Datasets/moseg_Jpglist_'+exprset, 'w')
classList=os.listdir(dspath)
for action in classList:
    vidlist=os.listdir(dspath+'/'+action)
    for img in vidlist:
        if img.endswith('jpg'):
            target.write(dspath+'/'+action+'/'+img)
            target.write("\n")
target.close()
print 'done'
