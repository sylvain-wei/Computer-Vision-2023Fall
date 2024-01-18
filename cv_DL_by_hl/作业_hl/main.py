import argparse 
import os
import random
from tqdm import tqdm

import torch 
import torch.nn as nn 
import torch.optim.lr_scheduler as lr_scheduler

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models


# 获取到具体的某一层layers
def get_module_layer(model: nn.Module, layer_name: str)->nn.Module:
    names = layer_name.split('_')
    if len(names) == 1:
        return model._modules.get(names[0])
    else:
        module = model
        for name in names:
            module = module._modules.get(name)
        return module

# 计算图像范数
def img_norm(x, alpha=2.0):
    return torch.abs(x ** alpha).sum()

# 计算图像正则化
def reg_img_prior(x, alpha=6.0):
    return img_norm(x, alpha)

# 计算全变分TV_norm
def reg_TV(x, beta=2.0):
    assert(x.size(0) == 1)
    image = x[0]
    dy = torch.zeros_like(image)
    dx = torch.zeros_like(image)
    dy[:, 1:, :] = image[:, :-1, :] - image[:, 1:, :]
    dx[:, :, 1:] = image[:, :, 1:] - image[:, :, :-1]
    return ((dx ** 2 + dy ** 2) ** (beta / 2.0)).sum()

# 计算归一化的重建损失
def rec_loss(x, ref_img):
    return img_norm(x-ref_img, alpha=2.0) / img_norm(ref_img, alpha=2.0)

def main(ori_img, model_name='alexnet', layer_name='feature_3', input_size=227,
         alpha=6.0, beta=2.0, lambda_alpha=1e-5, lambda_tv=1e-5,
         epochs=200, lr=1e2, momentum=0.9, stdout_epoches=20,
         decay_factor=0.1, decay_epochs=100, device='cuda'):
    
    # 进行图像预处理
    # 归一化参数
    # 大规模训练数据集的各个通道均值方差
    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]

    # 输入原图的转换
    transform = transforms.Compose([
        transforms.Resize(size=input_size), # 修改图片大小
        transforms.CenterCrop(size=input_size), # 中心剪裁
        transforms.ToTensor(),      # 图片转换为张量
        transforms.Normalize(mu, sigma),    # 归一化
    ])

    # 输出图像的转换
    # 需要进行反归一化，使得可视化效果能够更好
    def clip(tensor):
        return torch.clamp(tensor, 0, 1)
    
    # 反归一化
    detransform = transforms.Compose([
        transforms.Normalize(
            mean=[-m/s for m, s in zip(mu, sigma)],
            std=[1/s for s in sigma]),
        transforms.Lambda(clip),
        transforms.ToPILImage(),
    ])

    # 模型载入
    model = models.__dict__[model_name](pretrained=True)
    # 注意，这里不更新模型本身的梯度，我们只关心输入的白噪声图如何更新到目标图上去
    model.eval()
    model.to(device)

    # 原始图的预处理
    ref_img = transform(Image.open(ori_img)).unsqueeze(0)

    # 当前的激活特征值
    activations = []

    # TODO:考虑修改，是否不需要使用list
    def hook_activations(module, input, output):
        activations.append(output)
    
    def get_activations(model, input):
        del activations[:]
        _ = model(input)
        assert(len(activations) == 1)
        return activations[0]
    
    # 注册钩子函数，能够得到给定层的输出Tensor
    handle = get_module_layer(model, layer_name).register_forward_hook(hook_activations)
    
    ref_img.to(device)
    ref_img.requires_grad = False   # 禁止梯度反传
    ref_actavations = get_activations(model, ref_img).detach()  # 原始图reference的激活表征

    # 白噪声图，需要反传更新
    x = torch.nn.Parameter(1e-3 * torch.randn_like(ref_img, device=device), requires_grad=True)
    # 优化器：使用带动量的优化器，且优化目标仅为白噪声图
    optimizer = torch.optim.SGD([x], lr=lr, momentum=momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_epochs, gamma=decay_factor)


    for i in tqdm(range(epochs), total=epochs):
        x_activations = get_activations(model, x)

        # 重建损失、图像先验正则化、TV正则化
        loss = rec_loss(x_activations, ref_actavations)
        reg_alpha = reg_img_prior(x, alpha)
        reg_tv = reg_TV(x, beta)
        
        # 总的加权损失
        total_loss = loss + lambda_alpha * reg_alpha + lambda_tv * reg_tv

        if (i+1) % stdout_epoches == 0:
            print('Epoch %d:\t Reg Alpha: %f\tReg TV: %f\tRecon Loss: %f\tTotal Loss: %f'
                  % (i+1, reg_alpha.item(), reg_tv.item(), loss.item(), total_loss.item()))
        
        # 优化
        optimizer.zero_grad()
        total_loss.backward()   # 反传计算梯度
        optimizer.step()    # 更新参数

        scheduler.step()    # 优化器更新学习率

    recon_img = detransform(x.squeeze().data).convert("RGB")
    recon_img.show()
    recon_img.save(f'{model_name}_{layer_name}.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,   # 输入的图片相对路径
                        default='./reference_images/my_face.jpg')
    parser.add_argument('--model', type=str, default='alexnet') # 使用的CNN模型
    parser.add_argument('--input_size', type=int, default=227)  # 输入的图像大小
    parser.add_argument('--layer_name', type=str, 
                        default='avgpool')  # 选择反转特征的层名
    # 几个损失函数的超参数
    parser.add_argument('--alpha', type=float, default=6.0) # 图像先验正则化项超参
    parser.add_argument('--beta', type=float, default=2.0)  # 全变分正则化项超参
    parser.add_argument('--lambda_alpha', type=float, default=1e-5) # 图像先验正则化损失权重
    parser.add_argument('--lambda_tv', type=float, default=1e-5)    # 全变分正则化损失权重
    # 几个训练过程的超参数
    parser.add_argument('--epochs', type=int, default=200)  # 训练轮数，可能在100-200
    parser.add_argument('--lr', type=int, default=1e2)    # 学习率
    parser.add_argument('--momentum', type=float, default=0.9)  # SGD动量
    # 几个训练技巧
    parser.add_argument('--stdout_epoches', type=int, default=25)   # 每次打印的间隔轮数
    parser.add_argument('--decay_factor', type=float, default=0.1)
    parser.add_argument('--decay_epochs', type=int, default=100)    # 每隔多少次lr进行衰减
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=2023)   # 随机种子
    
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    # 设定种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    


    # 主函数
    main(ori_img=args.image_path, model_name=args.model, layer_name=args.layer_name,
         input_size=args.input_size, alpha=args.alpha, beta=args.beta, 
         lambda_alpha=args.lambda_alpha, lambda_tv=args.lambda_tv, epochs=args.epochs,
         lr=args.lr, momentum=args.momentum, stdout_epoches=args.stdout_epoches,
         decay_factor=args.decay_factor, decay_epochs=args.decay_epochs, device=args.device)