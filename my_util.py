import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
cpu = torch.device('cpu')


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])

            
def plot_conflict(grad1=torch.randn(10),grad2=torch.randn(10),base=20,
                  t1='task1',t2='task2',center=[5,0],L=np.array([0, 2]), figsize=(5, 5)):
    grad1,grad2=torch.flatten(grad1),torch.flatten(grad2)
    
    if grad1.shape!=grad2.shape:
        return print("Shape mismatch between two gradients")
    rad=torch.arccos(torch.dot(grad1,grad2)/(torch.norm(grad1)*torch.norm(grad2)))
    angle=torch.rad2deg(rad)
    fig, ax = plt.subplots(figsize=figsize,frameon=False)
    kw = dict(size=75, unit="points", text=str(int(angle.item()))+r"$°$")
    ax.set_aspect(1)

    phi = np.deg2rad(base)
    x = center[0] + np.cos(phi) * L
    y = center[1] + np.sin(phi) * L
    l1=ax.plot(x, y,linewidth=3,color='blue')
    ax.arrow(x[1],y[1],np.cos(phi)*0.01, np.sin(phi)*0.01, shape='full', lw=5,
       length_includes_head=True, head_width=.01,color='blue',label=t1)
    ax.text(x[1]-x[1]/6,y[1],t1)
    phi = np.deg2rad(angle.item()+base)
    x = center[0] + np.cos(phi) * L
    y = center[1] + np.sin(phi) * L
    ax.arrow(x[1],y[1],np.cos(phi)*0.01, np.sin(phi)*0.01, shape='full', lw=5,
       length_includes_head=True, head_width=.01,color='red',label=t2)
    ax.text(x[1],y[1],t2)
    l2=ax.plot(x, y,linewidth=3,color='red')
    AngleAnnotation(center, (l1[0].get_xdata()[1],l1[0].get_ydata()[1]), 
                    (l2[0].get_xdata()[1],l2[0].get_ydata()[1]), ax=ax, 
                    text_kw=dict(bbox=dict(boxstyle="round", fc="w")),**kw)
    ax.set_xticks([])
    ax.set_yticks([])
    lab='conflict'
    if int(angle.item())<=90:
        lab='no conflict'
    ax.set_xlabel(lab,fontsize=15)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.show()


    

            
def DM_rate(output,target,x_control):
    prot_att=x_control
    index_prot=torch.squeeze(torch.nonzero(prot_att[:] != 1.))
    target_prot=torch.index_select(target, 0, index=index_prot)
    index_prot_pos=torch.squeeze(torch.nonzero(target_prot[:] == 1. ))
    index_prot_neg=torch.squeeze(torch.nonzero(target_prot[:] == 0. ))

    index_non_prot=torch.squeeze(torch.nonzero(prot_att[:] == 1.))
    target_non_prot=torch.index_select(target, 0, index=index_non_prot)
    index_non_prot_pos=torch.squeeze(torch.nonzero(target_non_prot[:] == 1. ))
    index_non_prot_neg=torch.squeeze(torch.nonzero(target_non_prot[:] == 0. ))

    if index_prot_pos.shape==torch.Size([]) or index_prot_pos.shape==torch.Size([0])\
        or index_non_prot_pos.shape==torch.Size([]) or index_non_prot_pos.shape==torch.Size([0]):
            l_prot_pos=torch.tensor(0.0001)
            l_non_prot_pos=torch.tensor(0.0001)
    else:        
            l_prot_pos=acc(torch.index_select(output, 0, index=index_prot_pos),torch.index_select(target, 0, index=index_prot_pos))    
            l_non_prot_pos=acc(torch.index_select(output, 0, index=index_non_prot_pos),torch.index_select(target, 0, index=index_non_prot_pos))    
    
    if index_prot_neg.shape==torch.Size([]) or index_prot_neg.shape==torch.Size([0])\
        or index_non_prot_neg.shape==torch.Size([]) or index_non_prot_neg.shape==torch.Size([0]):
            l_prot_neg=torch.tensor(0.0001)
            l_non_prot_neg=torch.tensor(0.0001)
    else:        
            l_prot_neg=acc(torch.index_select(output, 0, index=index_prot_neg),torch.index_select(target, 0, index=index_prot_neg))    
            l_non_prot_neg=acc(torch.index_select(output, 0, index=index_non_prot_neg),torch.index_select(target, 0, index=index_non_prot_neg))  
            
    dl_pos=torch.abs(l_prot_pos-l_non_prot_pos)
    dl_neg=torch.abs(l_prot_neg-l_non_prot_neg)
    DM=dl_pos+dl_neg
    
    return DM, dl_pos

def fair_loss(output,target,x_control):
    prot_att=x_control
    index_prot=torch.squeeze(torch.nonzero(prot_att[:] != 1.))
    target_prot=torch.index_select(target, 0, index=index_prot)
    index_prot_pos=torch.squeeze(torch.nonzero(target_prot[:] == 1. ))
    index_prot_neg=torch.squeeze(torch.nonzero(target_prot[:] == 0. ))

    index_non_prot=torch.squeeze(torch.nonzero(prot_att[:] == 1.))
    target_non_prot=torch.index_select(target, 0, index=index_non_prot)
    index_non_prot_pos=torch.squeeze(torch.nonzero(target_non_prot[:] == 1. ))
    index_non_prot_neg=torch.squeeze(torch.nonzero(target_non_prot[:] == 0. ))

    l_prot_pos=F.cross_entropy(torch.index_select(output, 0, index=index_prot_pos),torch.index_select(target, 0, index=index_prot_pos))    
    l_non_prot_pos=F.cross_entropy(torch.index_select(output, 0, index=index_non_prot_pos),torch.index_select(target, 0, index=index_non_prot_pos))    
    l_non_prot_neg=F.cross_entropy(torch.index_select(output, 0, index=index_non_prot_neg),torch.index_select(target, 0, index=index_non_prot_neg))
    l_prot_neg=F.cross_entropy(torch.index_select(output, 0, index=index_prot_neg),torch.index_select(target, 0, index=index_prot_neg))    

    for l in [l_prot_pos,l_non_prot_pos,l_prot_neg,l_non_prot_neg]:
        if torch.isinf(l)==True:
            l=torch.zeros_like(l,requires_grad=True)
    dl_pos=torch.max(l_prot_pos,l_non_prot_pos)
    dl_neg=torch.max(l_prot_neg,l_non_prot_neg)
    L=dl_pos+dl_neg
    
    return L

def Update_model(model,grads_sh,omega,G_n,r_t,opti,paths):
    lr=0.001
    loss_gn=[(G_n[t]-torch.mean(G_n)*r_t[t]) for t in range(len(G_n))]
    for i in range(len(G_n)):
        d_l=0
        if loss_gn[i]>0:
            d_l+=(len(G_n)-1)/len(G_n)*G_n[i]
        elif loss_gn[i]<0:
            d_l-=(len(G_n)-1)/len(G_n)*G_n[i]
        for j in range(len(G_n)):
            if j!=i:
                if loss_gn[j]>0:
                    d_l-=(G_n[i]/len(G_n))
                elif loss_gn[j]<0:
                    d_l+=(G_n[i]/len(G_n))
        
        omega[i]-=lr*d_l

    
    for n,p in model.named_parameters():
        if p.data.shape[0]!=2 and p.grad==None:
            t_g,w_s=[],0
            for t in paths:
                flag,w=0,1
                if n.startswith('branches'):
                    br=n.split('.')[1]
                    if br in paths[t]:
                        flag=1
                elif n.startswith('shared'):
                    flag=1
                if flag==1:
                    t_g.append(t)
                    w_s+=omega[t]
            for i in range(len(t_g)):
                if i==0:
                    p.grad=(omega[t_g[i]]/w_s)*grads_sh[t_g[i]][n]
                else:
                    p.grad+=(omega[t_g[i]]/w_s)*grads_sh[t_g[i]][n]
                    
    opti.step() 
    total=sum(omega)
    for i in range(len(omega)):
        omega[i]=omega[i]/total
    return omega,model




import torch
import numpy as np
from CKA import CKA, CudaCKA
from copy import deepcopy
if torch.cuda.is_available():
    cka_dv='cuda:0'
else:
    cka_dv='cpu'
    
cka = CudaCKA(cka_dv)

def sim_mat(self,out=1,groups=[]):
    if isinstance(self,nn.DataParallel):
        self=self.module
    if out!=1:
        task_layers=list(self.named_children())[-2][1]
        k=0
        for task in task_layers:
            if task.startswith(str(out-1)):
                k+=1
        sim_mat=torch.zeros(k,k).to(cka_dv)
    else:
        task_layers=list(self.named_children())[-1][1]
        sim_mat=torch.zeros(len(task_layers),len(task_layers)).to(cka_dv)
    task_layers=task_layers.to(cka_dv)
    
    i=0
    for t1 in task_layers:
        v1=None
        if out!=1:
            if t1.startswith(str(out-1)):
                v1=deepcopy(task_layers[t1].weight.data.T)
        else:
            v1=deepcopy(task_layers[t1].weight.data.T)
        
        if v1!=None:
            j=0
            for t2 in task_layers:
                v2=None
                if out!=1:
                    if t2.startswith(str(out-1)):
                        v2=deepcopy(task_layers[t2].weight.data.T)
                else:
                    v2=deepcopy(task_layers[t2].weight.data.T)
                #print(i,j)
                if v2!=None:
                    if i==j:
                        sim_mat[i][j]=0
                        #sim_mat[j][i]=1
                    else:
                        if sim_mat[i][j]==0:
                            if len(groups)==0: 
                                sim_nets=cka.linear_CKA(v1, v2)
                                #print(sim_nets)
                                sim_mat[i][j]=sim_nets
                                sim_mat[j][i]=sim_nets
                            else:
                                flag=0
                                for g in range(len(groups)):
                                    if i in groups[g] and j in groups[g]:
                                        sim_nets=cka.linear_CKA(v1, v2)
                                        sim_mat[i][j]=deepcopy(sim_nets)
                                        sim_mat[j][i]=deepcopy(sim_nets)
                                        flag=1
                    j+=1
            i+=1
    return sim_mat


def find_groups(scores,method='agglomerative', tau=None):
    if method=='agglomerative':
        a,b=torch.sort(-scores)
        a=-a
        if tau==None:
            tau=torch.mean(a)
        #if group==None:
        group=[]
        for j in range(len(a)):
            flag=True
            for g in range(len(group)):
                if j in group[g]:
                    flag=False
            if flag:
                    #print(flag)
                    group.append([j])
            #print(group)
            for i in range(len(a)):
                #print(a[i][1],j)
                if i!=j:
                    if b[i][1]==j and b[j][1]==i and a[i][j]>=tau:
                        for g in range(len(group)):
                            if j in group[g] and i not in group[g]:
                                #print(j,i)
                                group[g].append(i)

    if method=='MST':
        a,b=torch.sort(-torch.unique(torch.flatten(scores)))
        a=-a
        if tau==None:
            tau=torch.mean(a)
        group=[]
        c=a[0]
        flag=True
        while(flag):
            if c<tau or sum([len(g) for g in group])==len(scores):
                flag=False
                break
            else:
                found=False
                for j in range(len(scores)):
                    if c in scores[j] and c<0.999:
                        r=j
                        v=(scores[j]==c).nonzero().item()
                        found=True
                        #print(r,v,group)
                        break
                if found:
                    present,k,m=0,-1,-1     
                    #print(r,v,group)
                    for g in range(len(group)):         
                        if r in group[g] and v not in group[g]:
                            present=1
                            k=g
                        if v in group[g] and r not in group[g]:
                            present=2
                            m=g
                        if v in group[g] and r in group[g]:
                            present=3
                    
                        
                    if k!=-1 and m!=-1:
                        present=-1
                    if present==0:
                        group.append([r,v])
                    elif present==1:
                        s=[1 if (scores[v][j]>=tau).item()==True else 0 for j in group[k]]
                        if sum(s)==len(group[k]):
                            group[k].append(v)
                    elif present==2:
                        s=[1 if (scores[r][j]>=tau).item()==True else 0 for j in group[m]]
                        if sum(s)==len(group[m]):
                            group[m].append(r)
                a=a[1:]
                c=a[0].item()
        for j in range(len(scores)):
            flag=True
            for g in group:
                if j in g:
                    flag=False
                    break
            if flag:
                group.append([j])
    return group


import copy
def branches(net=None,group=None,out=1,parents={},branches=None,premod=False):
    if group==None or isinstance(group, list)==False or len(parents)==0:
        print('Insufficient Arguments Error')
        return net
    else:
        groups=copy.deepcopy(group)
        if isinstance(net,nn.DataParallel):
            net=net.module
        if premod==True:
            pretrain=copy.deepcopy(net.pretrain.to(cpu))
            pretrain.fc.requires_grad=True
        new_parents=copy.deepcopy(parents)
        if branches==None:
            branches=nn.ModuleDict()    
        if out!=1:
            shared_layers =copy.deepcopy(net.shared[:-3].to(cpu))
            pf=net.shared[-3].to(cpu)
        else:
            shared_layers =copy.deepcopy(net.shared[:-1].to(cpu))
            pf=net.shared[-1].to(cpu)   
            
        task_layers=nn.ModuleDict({ch: copy.deepcopy(net.tasks_out[ch].to(cpu)) for ch in net.tasks_out})
        for i in range(len(groups)):
            mod=nn.Linear(pf.in_features,pf.out_features)
            mod.weight.data=pf.weight.data.detach().clone()
            mod.weight.requires_grad=True
            mod.bias.data=pf.bias.data.detach().clone()
            mod.bias.requires_grad=True
            branches[str(out)+str(i)]=mod
        del net
        if out!=1: 
            for i in range(len(new_parents)):
                for j in range(len(groups)):
                    key=int(''.join(new_parents[i][-1])[1:])#key=int(''.join(new_parents[i])[-1])
                    if key in groups[j]:           
                        new_parents[i].append(str(out)+str(j))
                        break
        else:
            for i in range(len(new_parents)):
                for j in range(len(groups)):
                    if i in groups[j]:
                        new_parents[i].append(str(out)+str(j)) 
                        break

        if premod==True:
            return pretrain,shared_layers,branches,task_layers,new_parents
        else:
            return shared_layers,branches,task_layers,new_parents

from copy import deepcopy        
def fair_grads(model,outputs,labels,xc,grouped,dv):
    def vector_rejection(g1,g2):
        v1=deepcopy(g1)
        v2=deepcopy(g2)
        return v1-(torch.dot(torch.flatten(v1),torch.flatten(v2))*v2/torch.square(torch.norm(v2)))
    if isinstance(model,nn.DataParallel):
            model=model.module
    for t in range(len(labels)):
        loss_f=fair_loss(outputs[t], labels[t].to(dv),xc.to(dv))
        loss_f.backward(retain_graph=True)
    if len(grouped)>0:
        for i in grouped:
            for gp in range(len(grouped[i])):
                if len(grouped[i][gp])>1:
                    if i!=1:
                        g1w=copy.deepcopy(model.branches[str(i-1)+str(grouped[i][gp][0])].weight.grad)
                        g1b=copy.deepcopy(model.branches[str(i-1)+str(grouped[i][gp][0])].bias.grad)
                        g2w=copy.deepcopy(model.branches[str(i-1)+str(grouped[i][gp][1])].weight.grad)
                        g2b=copy.deepcopy(model.branches[str(i-1)+str(grouped[i][gp][1])].bias.grad)

                        if torch.dot(torch.flatten(g1w),torch.flatten(g2w))<0:
                            model.branches[str(i-1)+str(grouped[i][gp][0])].weight.grad=deepcopy(vector_rejection(g1w,g2w))
                            model.branches[str(i-1)+str(grouped[i][gp][0])].bias.grad=deepcopy(vector_rejection(g1b,g2b))
                            model.branches[str(i-1)+str(grouped[i][gp][1])].weight.grad=deepcopy(vector_rejection(g2w,g1w))
                            model.branches[str(i-1)+str(grouped[i][gp][1])].bias.grad=deepcopy(vector_rejection(g2b,g1b))
                    else:
                        g1w=copy.deepcopy(model.tasks_out['ch'+str(grouped[i][gp][0])].weight.grad)
                        g1b=copy.deepcopy(model.tasks_out['ch'+str(grouped[i][gp][0])].bias.grad)
                        g2w=copy.deepcopy(model.tasks_out['ch'+str(grouped[i][gp][1])].weight.grad)
                        g2b=copy.deepcopy(model.tasks_out['ch'+str(grouped[i][gp][1])].bias.grad)

                        if torch.dot(torch.flatten(g1w),torch.flatten(g2w))<0:
                            model.tasks_out['ch'+str(grouped[i][gp][0])].weight.grad=deepcopy(vector_rejection(g1w,g2w))
                            model.tasks_out['ch'+str(grouped[i][gp][0])].bias.grad=deepcopy(vector_rejection(g1b,g2b))
                            model.tasks_out['ch'+str(grouped[i][gp][1])].weight.grad=deepcopy(vector_rejection(g2w,g1w))
                            model.tasks_out['ch'+str(grouped[i][gp][1])].bias.grad=deepcopy(vector_rejection(g2b,g1b))

    return model



from copy import deepcopy
def get_grads(model):   
    if isinstance(model,nn.DataParallel):
        model=model.module
    grads_t={}
    for n,p in model.named_parameters():
        if p.grad!=None and n.split('.')[-1]!='bias' and n.split('.')[-2].startswith('bn')!=True and p.data.shape[0]!=2 and p.requires_grad==True and torch.count_nonzero(p.grad)>0:
                grads_t[n.split('.')[1]] = deepcopy(torch.flatten(p.grad))
        p.grad=None
        #p.detach()
        #p.requires_grad=True
        del p.grad
        #p.grad.zero_()
    return grads_t,model

def get_task_heats(task_grads=[{}],task_heats=torch.zeros(1,1)):
    if task_heats.shape.numel()!=len(task_grads)**2:
        print("invalid argment-->heats must be a square matrix of dim len(grads)")
        return task_heats
    else:
        for i in range(len(task_grads)):
            for j in range(i+1,len(task_grads)):
                for n in task_grads[i]:
                    if n in task_grads[j]:
                        if torch.dot(task_grads[i][n],task_grads[j][n])<0:
                            task_heats[i][j]+=1
                            task_heats[j][i]+=1
        return task_heats
        
def _get_params(self):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.data
        return params
def get_flat_params(self,dv):
        """Get flattened and concatenated params of the model."""
        params = _get_params(self)
        flat_params = torch.Tensor().to(dv)
        #if torch.cuda.is_available() and dv==torch.device('cuda'):
        #    flat_params = flat_params.cuda()
        for _, param in params.items():
            flat_params = torch.cat((flat_params, torch.flatten(param)))
        return flat_params
def _get_param_shapes(model):
        shapes = []
        for name, param in model.named_parameters():
            shapes.append((name, param.shape, param.numel()))
        return shapes
    
    
    
def get_conflicts(task_grads,N_tasks):
    def cal_conf(v1,v2):
        return torch.rad2deg(torch.arccos(torch.dot(v1,v2)/(torch.norm(v1)*torch.norm(v2)))).to(cpu)

    cnf_scores=[]
    for t1 in range(N_tasks):
        d1=task_grads[t1]
        if (t1+1)<N_tasks:
            for t2 in range(t1+1,N_tasks):
                d2=task_grads[t2]
                cnf_1_2=[cal_conf(d1[v],d2[v]) for v in d1 if v in d2]
                cnf_scores.append(cnf_1_2)
    return cnf_scores