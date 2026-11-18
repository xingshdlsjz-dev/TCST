import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

class TCST(nn.Module):
    def __init__(self, channels, classes, action_mode='ground_truth', d=128, dropout=.2):
        super().__init__(); self.mode=action_mode
        self.encoder=nn.Sequential(nn.Conv1d(channels,d,5,padding=2,bias=False),nn.BatchNorm1d(d),nn.GELU(),nn.Conv1d(d,d,5,padding=2,bias=False),nn.BatchNorm1d(d),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        layer=nn.TransformerEncoderLayer(d,4,4*d,dropout,activation='gelu',batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2)
        self.action=None if action_mode=='independent' else nn.Embedding(classes,d)
        self.transition=nn.Sequential(nn.Linear(d if action_mode=='independent' else 2*d,d),nn.GELU(),nn.Dropout(dropout),nn.Linear(d,d))
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Dropout(dropout),nn.Linear(d,classes))
    def forward(self,x,y=None):
        b,t,l,c=x.shape; z=self.encoder(x.reshape(b*t,l,c).transpose(1,2)).squeeze(-1).reshape(b,t,-1)
        mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),1); h=self.temporal(z,mask=mask); logits=self.head(h)
        if t<2:return logits,None
        if self.mode=='independent': u=h[:,:-1]
        elif self.mode=='ground_truth': u=torch.cat((h[:,:-1],self.action(y[:,:-1])),dim=-1)
        else: u=torch.cat((h[:,:-1],torch.softmax(logits[:,:-1],-1).detach()@self.action.weight),dim=-1)
        loss=nn.functional.mse_loss(self.transition(u),h[:,1:].detach())
        return logits,loss

class HARMamba(nn.Module):
    """Bidirectional selective-gated state-space baseline; pure PyTorch implementation."""
    def __init__(self,channels,classes,d=128,dropout=.2):
        super().__init__(); self.proj=nn.Linear(channels*2,d)
        self.in_gate=nn.Linear(d,2*d); self.decay=nn.Parameter(torch.zeros(d)); self.out=nn.Linear(d,d)
        self.norm=nn.LayerNorm(d); self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(d,classes))
    def scan(self,x,reverse=False):
        if reverse:x=x.flip(1)
        val,gate=self.in_gate(x).chunk(2,-1); gate=torch.sigmoid(gate); decay=torch.sigmoid(self.decay).view(1,-1)
        state=torch.zeros_like(val[:,0]); out=[]
        for i in range(val.shape[1]):
            state=decay*state+(1-decay)*val[:,i]*gate[:,i]; out.append(state)
        result=torch.stack(out,1)
        return result.flip(1) if reverse else result
    def forward(self,x):
        # Window mean/dispersion summaries preserve identical 128x27 input evidence.
        u=self.proj(torch.cat([x.mean(2),x.std(2,unbiased=False)],-1))
        h=self.norm(self.out(self.scan(u)+self.scan(u,True))); return self.head(h)

def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1] / 'data' / 'subject108_split.npz');p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=['tcst','harmamba'],default='tcst');p.add_argument('--action-mode',choices=['ground_truth','independent','predicted'],default='ground_truth');p.add_argument('--epochs',type=int,default=15);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    seed_all(a.seed); data=np.load(a.data,allow_pickle=False); meta=json.loads(str(data['metadata'])); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); k=len(meta['activity_ids'])
    def loader(split,shuffle): return DataLoader(TensorDataset(torch.tensor(data['x_'+split],dtype=torch.float32),torch.tensor(data['y_'+split],dtype=torch.long)),batch_size=64,shuffle=shuffle)
    tr,va,te=[loader(s,s=='train') for s in ('train','val','test')]
    model=(TCST(27,k,a.action_mode) if a.model=='tcst' else HARMamba(27,k)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); ce=nn.CrossEntropyLoss(label_smoothing=.02); best=-1; a.out.mkdir(parents=True,exist_ok=True)
    for ep in range(1,a.epochs+1):
        model.train()
        for x,y in tr:
            x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True)
            if a.model=='tcst': logits,tl=model(x,y); loss=ce(logits.reshape(-1,k),y.reshape(-1))+.2*tl
            else: logits=model(x); loss=ce(logits.reshape(-1,k),y.reshape(-1))
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        model.eval();ys=[];ps=[]
        with torch.no_grad():
            for x,y in va:
                x,y=x.to(device),y.to(device); logits=model(x,y)[0] if a.model=='tcst' else model(x);ys.extend(y.flatten().cpu().tolist());ps.extend(logits.argmax(-1).flatten().cpu().tolist())
        vf=f1_score(ys,ps,average='macro',zero_division=0);print(f'epoch={ep} val_acc={accuracy_score(ys,ps):.5f} val_macro_f1={vf:.5f}',flush=True)
        if vf>best: best=vf;torch.save(model.state_dict(),a.out/'best_model.pt');best_ep=ep
    model.load_state_dict(torch.load(a.out/'best_model.pt',map_location=device,weights_only=True));model.eval();ys=[];ps=[]
    with torch.no_grad():
        for x,y in te:
            x,y=x.to(device),y.to(device);logits=model(x,y)[0] if a.model=='tcst' else model(x);ys.extend(y.flatten().cpu().tolist());ps.extend(logits.argmax(-1).flatten().cpu().tolist())
    result={'model':a.model,'action_mode':a.action_mode if a.model=='tcst' else None,'seed':a.seed,'best_epoch':best_ep,'test_windows':len(ys),'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro',zero_division=0)}
    (a.out/'metrics.json').write_text(json.dumps(result,indent=2));(a.out/'classification_report.txt').write_text(classification_report(ys,ps,target_names=meta['activity_names'],zero_division=0));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
