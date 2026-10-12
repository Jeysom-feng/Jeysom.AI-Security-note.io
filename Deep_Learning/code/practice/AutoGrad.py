import torch

w=torch.tensor(10, requires_grad=True,dtype=float)

for i in range(100):
    #损失函数
    loss=4*w**2
    #重置梯度
    if(w.grad is not None):
        w.grad.zero_()
    #自动微分算出梯度
    loss.sum().backward()
    #更新参数
    w.data=w.data-0.01*w.grad
    print(f"参数w.data: {w.data}, 梯度w.grad: {w.grad}")