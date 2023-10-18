import numpy as np
import os
os.chdir("/yourpath to CelebA data folder/CelebA")
# use the link "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" to get download instruction
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
cpu = torch.device('cpu')#"cuda:0" if torch.cuda.is_available() else "cpu")
dv=torch.device('cuda')#torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_celeba(protected='Age',batch_size=8192,mode='train',return_ft=False):
    f=open('/yourpath to CelebA data folder/CelebA/list_attr_celeba.txt').read()
    Ft=f.split('\n')[1].split()
    X=np.array([r.split() for r in f.split('\n')[2:-1]])
    ids=X[:,0]
    
    if protected=='Age':
        g=X[:,Ft.index('Young')+1]
        y=np.array([X[:,i+1] for i in range(len(Ft)) if i!=Ft.index('Young')])
        y=[[0 if t[i]=='-1' else 1 for i in range(len(t))] for t in y]
        g=np.array([0 if g[i]=='-1' else 1 for i in range(len(g))])
        no_tasks=[4,10,17,13,14,17,26,29,35]
        y=np.array([y[t] for t in range(len(y)) if t not in no_tasks])
        Ft.remove('Young')
        Features=[Ft[t] for t in range(len(Ft)) if t not in no_tasks]
    elif protected=='Gender':
        g=X[:,Ft.index('Male')+1]
        y=np.array([X[:,i+1] for i in range(len(Ft)) if i!=Ft.index('Male')])
        y=[[0 if t[i]=='-1' else 1 for i in range(len(t))] for t in y]
        g=np.array([0 if g[i]=='-1' else 1 for i in range(len(g))])
        tasks=[2,3,5,6,7,8,11,12,19,20,22,24,26,30,31,32,38]
        y=np.array([y[t] for t in tasks])
        Ft.remove('Male')
        Features=[Ft[t] for t in tasks]
        
    if return_ft:
        return Features
    else:
        dataroot = "/home/roy/CelebA/celeba/"
        image_size = 64
        batch_size = batch_size
        workers = 2

        dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        indices=np.arange(len(ids))
        in_tr,in_val,in_ts=indices[:162770],indices[162770:182637],indices[182637:]
        in_tr1=in_tr[:5*batch_size]; in_tr2=in_tr[5*batch_size:10*batch_size]; in_tr3=in_tr[10*batch_size:15*batch_size]
        in_tr4=in_tr[15*batch_size:]
        in_val1,in_val2=in_val[:10000],in_val[10000:]
        if mode=='train':
            dataset_tr1 = torch.utils.data.Subset(dataset, in_tr1)
            dataset_tr2 = torch.utils.data.Subset(dataset, in_tr2)
            dataset_tr3 = torch.utils.data.Subset(dataset, in_tr3)
            dataset_tr4 = torch.utils.data.Subset(dataset, in_tr4)
            dataset_val1 = torch.utils.data.Subset(dataset, in_val1)
            dataset_val2 = torch.utils.data.Subset(dataset, in_val2)
            

            dataloader1 = torch.utils.data.DataLoader(dataset_tr1, batch_size=batch_size,
                                                     shuffle=False, num_workers=workers)
            dataloader2 = torch.utils.data.DataLoader(dataset_tr2, batch_size=batch_size,
                                                     shuffle=False, num_workers=workers)
            dataloader3 = torch.utils.data.DataLoader(dataset_tr3, batch_size=batch_size,
                                                     shuffle=False, num_workers=workers)
            dataloader4 = torch.utils.data.DataLoader(dataset_tr4, batch_size=batch_size,
                                                     shuffle=False, num_workers=workers)

            dataloader_val1 = torch.utils.data.DataLoader(dataset_val1, batch_size=10000,
                                                     shuffle=False, num_workers=workers)
            dataloader_val2 = torch.utils.data.DataLoader(dataset_val2, batch_size=10000,
                                                     shuffle=False, num_workers=workers)

            


            

            y_v1,y_v2=[y[i][in_val1] for i in range(len(y))],[y[i][in_val2] for i in range(len(y))]

            y_tr1,y_tr2=[y[i][in_tr1] for i in range(len(y))],[y[i][in_tr2] for i in range(len(y))]
            y_tr3,y_tr4=[y[i][in_tr3] for i in range(len(y))],[y[i][in_tr4] for i in range(len(y))]



            g_tr1,g_tr2,g_tr3,g_tr4=g[in_tr1],g[in_tr2],g[in_tr3],g[in_tr4]
            g_val1,g_val2=g[in_val1],g[in_val2]
            
            N_tasks=len(y)#4#2#3

            d_v=[dataloader_val1,dataloader_val2]
            yv=[y_v1,y_v2]
            gv=[g_val1,g_val2]
            dt=[dataloader1,dataloader2,dataloader3,dataloader4]
            yt=[y_tr1,y_tr2,y_tr3,y_tr4]
            gt=[g_tr1,g_tr2,g_tr3,g_tr4]


            return dt,yt, gt, d_v, yv, gv, batch_size
        else:
            dataset_ts = torch.utils.data.Subset(dataset, in_ts)
            dataloader_test = torch.utils.data.DataLoader(dataset_ts, batch_size=len(in_ts),
                                                     shuffle=False, num_workers=workers)
            y_test=[y[i][in_ts] for i in range(len(y))]
            g_test=g[in_ts]
            
            return dataloader_test, y_test, g_test