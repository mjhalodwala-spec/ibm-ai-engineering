import argparse, torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def main():
 p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--data",default="data"); a=p.parse_args()
 torch.manual_seed(42); tfm=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.243,.261))])
 ds=datasets.CIFAR10(a.data,train=True,download=True,transform=tfm); train,val=random_split(ds,[45000,5000],generator=torch.Generator().manual_seed(42)); train_dl=DataLoader(train,batch_size=a.batch_size,shuffle=True); val_dl=DataLoader(val,batch_size=a.batch_size)
 class CNN(nn.Module):
  def __init__(self):
   super().__init__(); self.features=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.head=nn.Linear(128,10)
  def forward(self,x): return self.head(self.features(x).flatten(1))
 device="cuda" if torch.cuda.is_available() else "cpu"; model=CNN().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
 for epoch in range(a.epochs):
  model.train(); seen=correct=total_loss=0
  for x,y in train_dl:
   x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step(); seen+=len(y); correct+=(logits.argmax(1)==y).sum().item(); total_loss+=loss.item()*len(y)
  model.eval(); hits=n=0
  with torch.no_grad():
   for x,y in val_dl: x,y=x.to(device),y.to(device); hits+=(model(x).argmax(1)==y).sum().item(); n+=len(y)
  print(f"epoch={epoch+1} train_loss={total_loss/seen:.4f} train_acc={correct/seen:.3f} val_acc={hits/n:.3f}")
 torch.save(model.state_dict(),"cifar10_cnn.pt")
if __name__=="__main__": main()
