from torchattacks.attacks.square import Square
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torchattacks
from CNN_ImgClassfier import imgclassifier
from torch.utils.data import DataLoader

#数据集创建
def creatDataset():
    train_dataset = CIFAR10(root="D:\\Desktop\\Python_file\\DL_Project\\data", train=True, download=False, transform=ToTensor())
    test_dataset = CIFAR10(root="D:\\Desktop\\Python_file\\DL_Project\\data", train=False,download=False,transform=ToTensor())
    return train_dataset, test_dataset


#APGD-CE对抗测试
def APGD_CE(model,test_dataset):
     model.eval()
     attack = torchattacks.APGD(
        model, 
        norm='Linf', 
        eps=8/255, 
        steps=10,
        loss='ce'
    )
     #生成对抗样本
     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
     total_clean = 0
     total_adv = 0
     total_samples = 0
     for imgs, labels in test_loader:
        with torch.no_grad():
            clean_logits = model(imgs)
        clean_pred = torch.argmax(clean_logits, dim=1)
        # 生成对抗样本
        adv_imgs = attack(imgs, labels)
        with torch.no_grad():
            adv_logits = model(adv_imgs)
        adv_pred = torch.argmax(adv_logits, dim=1)
        total_clean += (clean_pred == labels).sum().item()
        total_adv += (adv_pred == labels).sum().item()
        total_samples += imgs.shape[0]
     acc_clean = total_clean / total_samples
     acc_adv = total_adv / total_samples
     print(f"Clean accuracy: {acc_clean:.2%}")
     print(f"adv accuracy: {acc_adv:.2%}")

#APGD-DLR
def APGD_DLR(model,test_dataset):
    model.eval()
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
    total_sample,adv_correct,clean_correct=0,0,0
    #构造APGD-DLR对象
    attack = torchattacks.APGD(
            model, 
            norm='Linf', 
            eps=8/255, 
            steps=100,
            loss='dlr',
            n_restarts=3
        )
    for x,y in test_loader:
        x_adv = attack(x,y)
        with torch.no_grad():
            adv_pred =torch.argmax(model(x_adv),dim=1) 
        adv_correct += (adv_pred == y).sum().item()
        with torch.no_grad():
            clean_pred = torch.argmax(model(x),dim=1)
        clean_correct += (clean_pred == y).sum().item()
        total_sample += x.size(0)
        acc_clean = clean_correct / total_sample
        acc_adv = adv_correct / total_sample
    print(f"Clean accuracy: {acc_clean:.2%}")
    print(f"adv accuracy: {acc_adv:.2%}")

def FAB(model,test_dataset):
     model.eval()
     test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
     total_sample,adv_correct,clean_correct=0,0,0
        #构造FAB对象
     attack = torchattacks.FAB(
                model, 
                norm='Linf', 
                eps=8/255, 
                steps=15,
                multi_targeted=True 
            )
     for x,y in test_loader:
            x_adv = attack(x,y)
            with torch.no_grad():
                adv_pred =torch.argmax(model(x_adv),dim=1) 
            adv_correct += (adv_pred == y).sum().item()
            with torch.no_grad():
                clean_pred = torch.argmax(model(x),dim=1)
            clean_correct += (clean_pred == y).sum().item()
            total_sample += x.size(0)
            acc_clean = clean_correct / total_sample
            acc_adv = adv_correct / total_sample
     print(f"Clean accuracy: {acc_clean:.2%}")
     print(f"adv accuracy: {acc_adv:.2%}")

def Square_Attack(model,test_dataset):
         model.eval()
         test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)
         total_sample,adv_correct,clean_correct=0,0,0
            #构造SquareAttack对象
         attack =Square(
                    model, 
                    norm='Linf', 
                    eps=8/255, 
                    n_queries=10000,
                    n_restarts = 1
                )
         for x,y in test_loader:
                x_adv = attack(x,y)
                with torch.no_grad():
                    adv_pred =torch.argmax(model(x_adv),dim=1) 
                adv_correct += (adv_pred == y).sum().item()
                with torch.no_grad():
                    clean_pred = torch.argmax(model(x),dim=1)
                clean_correct += (clean_pred == y).sum().item()
                total_sample += x.size(0)
                acc_clean = clean_correct / total_sample
                acc_adv = adv_correct / total_sample
         print(f"Clean accuracy: {acc_clean:.2%}")
         print(f"adv accuracy: {acc_adv:.2%}")


     

#################################
if __name__ == "__main__":
     model = imgclassifier()
     model.load_state_dict(torch.load("D:\\Desktop\\Python_file\\DL_Project\\model\\CNNmodel.pth"))
     train_dataset, test_dataset = creatDataset()
    #  APGD_CE(model,test_dataset)
    #  APGD_DLR(model,test_dataset)
    #  FAB(model,test_dataset)
    # Square_Attack(model,test_dataset)