import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import MODEL_REGISTRY
from models.train_tcst import TCSTHAR

class HARMamba(nn.Module):
    def __init__(self, input_size, num_classes, d_model=128, dropout=.2, **kwargs):
        super().__init__(); d=d_model
        self.patch=nn.Conv1d(input_size,d,kernel_size=16,stride=8);self.mix=nn.Linear(d,2*d);self.decay=nn.Parameter(torch.zeros(d));self.out=nn.Linear(d,d);self.norm=nn.LayerNorm(d);self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(d,num_classes))
    def scan(self,x,reverse=False):
        if reverse:x=x.flip(1)
        v,g=self.mix(x).chunk(2,-1);g=torch.sigmoid(g);a=torch.sigmoid(self.decay).view(1,-1);s=torch.zeros_like(v[:,0]);ys=[]
        for i in range(v.shape[1]):s=a*s+(1-a)*v[:,i]*g[:,i];ys.append(s)
        y=torch.stack(ys,1);return y.flip(1) if reverse else y
    def forward(self,x):
        z=self.patch(x.transpose(1,2)).transpose(1,2);h=self.norm(self.out(self.scan(z)+self.scan(z,True)));return self.head(h.mean(1))

class ModernBlock(nn.Module):
    def __init__(self,d,k,drop):
        super().__init__();self.dw=nn.Conv1d(d,d,k,padding=k//2,groups=d,bias=False);self.bn=nn.BatchNorm1d(d);self.pw=nn.Sequential(nn.Conv1d(d,2*d,1),nn.GELU(),nn.Dropout(drop),nn.Conv1d(2*d,d,1));self.norm=nn.BatchNorm1d(d);self.scale=nn.Parameter(torch.ones(1,d,1)*0.1)
    def forward(self,x):return x+self.scale*self.norm(self.pw(torch.nn.functional.gelu(self.bn(self.dw(x)))))
class ModernTCN(nn.Module):
    def __init__(self,input_size,num_classes,d_model=128,dropout=.2,**kwargs):
        super().__init__();d=d_model;self.stem=nn.Sequential(nn.Conv1d(input_size,d,1,bias=False),nn.BatchNorm1d(d),nn.GELU());self.blocks=nn.Sequential(*[ModernBlock(d,k,dropout) for k in (7,15,31,63,7)]);self.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.LayerNorm(d),nn.Dropout(dropout),nn.Linear(d,num_classes))
    def forward(self,x):return self.head(self.blocks(self.stem(x.transpose(1,2))))

class SequenceDataset(Dataset):
    def __init__(self,x,y,groups,seq_len=6,tcst=False):
        self.x=x;self.y=y;self.seq_len=seq_len;self.tcst=tcst;self.starts=[]
        for g in np.unique(groups):
            idx=np.where(groups==g)[0]
            for s in range(int(idx[0]),int(idx[-1])-seq_len+2):self.starts.append(s)
    def __len__(self):return len(self.starts)
    def __getitem__(self,i):
        s=self.starts[i];xx=torch.from_numpy(self.x[s:s+self.seq_len].astype(np.float32));yy=torch.from_numpy(self.y[s:s+self.seq_len].copy()).long()
        return (xx,yy) if self.tcst else (xx.reshape(-1,xx.shape[-1]),yy[-1])

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def scores(y,p):return accuracy_score(y,p),f1_score(y,p,average='macro',zero_division=0)

def train_one(name,args,z,meta,device):
    tcst=name=='tcst_har'; loaders=[]
    for split in ('train','val','test'):
        ds=SequenceDataset(z['x_'+split],z['y_'+split],z['group_'+split],6,tcst)
        loaders.append(DataLoader(ds,batch_size=args.batch_size,shuffle=split=='train',num_workers=0,pin_memory=True))
    train,val,test=loaders;k=len(meta['classes']);channels=z['x_train'].shape[-1]
    if tcst:model=TCSTHAR(channels,k,latent_size=192,heads=8,layers=3,dropout=.15,action_mode='ground_truth')
    elif name=='harmamba':model=HARMamba(channels,k)
    elif name=='moderntcn':model=ModernTCN(channels,k,d_model=192,dropout=.15)
    elif name=='resnet1d':model=MODEL_REGISTRY[name](input_size=channels,num_classes=k,hidden_size=192,layers=5,dropout=.15)
    else:model=MODEL_REGISTRY[name](input_size=channels,num_classes=k,seq_len=6*meta['window'],hidden_size=128,d_model=128,heads=4,layers=2,patch_len=16,stride=8,dropout=.25)
    model=model.to(device);counts=np.bincount(z['y_train'],minlength=k).astype(np.float32);w=torch.tensor(counts.sum()/(k*np.maximum(counts,1)),device=device);criterion=nn.CrossEntropyLoss(weight=w,label_smoothing=.02);base_lr=1e-2 if name=='deepconvlstm' else 1e-3;opt=torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=5)
    out=args.output/name;out.mkdir(parents=True,exist_ok=True);best=-1;bad=0;history=[]
    for ep in range(1,args.epochs+1):
        model.train();losses=[]
        for x,y in train:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            if tcst:logits,tl=model(x,y);loss=criterion(logits[:,-1],y[:,-1])+.1*tl
            else:logits=model(x);loss=criterion(logits,y)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
        model.eval();ys=[];ps=[]
        with torch.no_grad():
            for x,y in val:
                x=x.to(device);logits=model(x,y.to(device))[0][:,-1] if tcst else model(x);target=y[:,-1] if tcst else y;ys.extend(target.tolist());ps.extend(logits.argmax(-1).cpu().tolist())
        va,vf=scores(ys,ps);sched.step(vf);history.append({'epoch':ep,'train_loss':float(np.mean(losses)),'val_accuracy':va,'val_macro_f1':vf,'learning_rate':opt.param_groups[0]['lr']});print(f'{name} epoch {ep:02d} val_acc={va:.4f} val_f1={vf:.4f} lr={opt.param_groups[0]["lr"]:.2e}',flush=True)
        if vf>best:best=vf;best_ep=ep;bad=0;torch.save({'model_state':model.state_dict(),'model':name,'metadata':meta,'args':vars(args)},out/'best_model.pt')
        else:bad+=1
    model.load_state_dict(torch.load(out/'best_model.pt',map_location=device,weights_only=False)['model_state']);model.eval();ys=[];ps=[]
    with torch.no_grad():
        for x,y in test:
            x=x.to(device);logits=model(x,y.to(device))[0][:,-1] if tcst else model(x);target=y[:,-1] if tcst else y;ys.extend(target.tolist());ps.extend(logits.argmax(-1).cpu().tolist())
    acc,f1=scores(ys,ps);res={'model':name,'best_epoch':best_ep,'test_samples':len(ys),'test_accuracy':acc,'test_macro_f1':f1,'parameters':sum(p.numel() for p in model.parameters())}
    (out/'metrics.json').write_text(json.dumps(res,indent=2));(out/'classification_report.txt').write_text(classification_report(ys,ps,target_names=meta['classes'],zero_division=0));np.savez_compressed(out/'predictions.npz',targets=np.asarray(ys),preds=np.asarray(ps));
    with (out/'history.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=history[0]);wr.writeheader();wr.writerows(history)
    print(json.dumps(res),flush=True);return res

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--models',default='tcst_har,cnn1d,deepconvlstm,lstm,patchtst,itransformer,resnet1d,inceptiontime,tcn,grufcn,timesnet,harmamba,moderntcn');p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--patience',type=int,default=5);p.add_argument('--seed',type=int,default=42);args=p.parse_args();seed_all(args.seed)
    z=np.load(args.data,allow_pickle=False);meta=json.loads(str(z['metadata']));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');args.output.mkdir(parents=True,exist_ok=True);results=[]
    for name in [x.strip() for x in args.models.split(',') if x.strip()]:results.append(train_one(name,args,z,meta,device));torch.cuda.empty_cache()
    with (args.output/'summary.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=results[0]);wr.writeheader();wr.writerows(results)
if __name__=='__main__':main()
