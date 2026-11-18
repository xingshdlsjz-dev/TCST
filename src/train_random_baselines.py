import argparse,json,random
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import accuracy_score,f1_score,classification_report

class HARMamba(nn.Module):
    """Bidirectional gated state-space HAR baseline, implemented in plain PyTorch."""
    def __init__(self,c,k,d=128,drop=.2):
        super().__init__(); self.proj=nn.Linear(c,d);self.inp=nn.Linear(d,2*d);self.decay=nn.Parameter(torch.zeros(d));self.out=nn.Linear(d,d);self.norm=nn.LayerNorm(d);self.head=nn.Sequential(nn.Dropout(drop),nn.Linear(d,k))
    def scan(self,x,rev=False):
        if rev:x=x.flip(1)
        a,g=self.inp(x).chunk(2,-1);g=torch.sigmoid(g);r=torch.sigmoid(self.decay).unsqueeze(0);s=torch.zeros_like(a[:,0]);o=[]
        for t in range(a.shape[1]):s=r*s+(1-r)*a[:,t]*g[:,t];o.append(s)
        y=torch.stack(o,1);return y.flip(1) if rev else y
    def forward(self,x):
        u=self.proj(x);return self.head(self.norm(self.out(self.scan(u)+self.scan(u,True))).mean(1))

class ModernBlock(nn.Module):
    def __init__(self,d,kernel,drop):
        super().__init__();self.dw=nn.Conv1d(d,d,kernel,padding=kernel//2,groups=d,bias=False);self.bn=nn.BatchNorm1d(d);self.pw=nn.Sequential(nn.Conv1d(d,2*d,1),nn.GELU(),nn.Dropout(drop),nn.Conv1d(2*d,d,1));self.norm=nn.BatchNorm1d(d)
    def forward(self,x):return x+self.norm(self.pw(torch.nn.functional.gelu(self.bn(self.dw(x)))))
class ModernTCN(nn.Module):
    def __init__(self,c,k,d=128,drop=.2):
        super().__init__();self.stem=nn.Conv1d(c,d,1);self.blocks=nn.Sequential(*[ModernBlock(d,ks,drop) for ks in (7,15,31,7)]);self.head=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.LayerNorm(d),nn.Dropout(drop),nn.Linear(d,k))
    def forward(self,x):return self.head(self.blocks(self.stem(x.transpose(1,2))))

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1] / 'data' / 'random_windows.npz');p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=['harmamba','moderntcn'],required=True);p.add_argument('--epochs',type=int,default=60);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 z=np.load(a.data,allow_pickle=False);meta=json.loads(str(z['metadata']));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');k=len(meta['activity_names'])
 def dl(s,sh):return DataLoader(TensorDataset(torch.tensor(z['x_'+s],dtype=torch.float32),torch.tensor(z['y_'+s],dtype=torch.long)),batch_size=128,shuffle=sh)
 tr,va,te=[dl(s,s=='train') for s in ('train','val','test')];model=(HARMamba(27,k) if a.model=='harmamba' else ModernTCN(27,k)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);crit=nn.CrossEntropyLoss(label_smoothing=.02);best=-1;a.out.mkdir(parents=True,exist_ok=True)
 for ep in range(1,a.epochs+1):
  model.train()
  for x,y in tr:
   x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);loss=crit(model(x),y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
  model.eval();ys=[];ps=[]
  with torch.no_grad():
   for x,y in va:ys.extend(y.tolist());ps.extend(model(x.to(device)).argmax(-1).cpu().tolist())
  f=f1_score(ys,ps,average='macro',zero_division=0);print(f'{a.model} epoch {ep:02d}: val_acc={accuracy_score(ys,ps):.5f}, val_macro_f1={f:.5f}',flush=True)
  if f>best:best=f;best_ep=ep;torch.save(model.state_dict(),a.out/'best_model.pt')
 model.load_state_dict(torch.load(a.out/'best_model.pt',map_location=device,weights_only=True));model.eval();ys=[];ps=[]
 with torch.no_grad():
  for x,y in te:ys.extend(y.tolist());ps.extend(model(x.to(device)).argmax(-1).cpu().tolist())
 res={'model':a.model,'protocol':'random-window; 8516/947/2366','seed':a.seed,'best_epoch':best_ep,'test_count':len(ys),'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro',zero_division=0)}
 (a.out/'metrics.json').write_text(json.dumps(res,indent=2));(a.out/'classification_report.txt').write_text(classification_report(ys,ps,target_names=meta['activity_names'],zero_division=0));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
