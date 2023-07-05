import torch
import torch.nn as nn
import yaml


class LogFunction(nn.Module):
    def forward(self, x):
        ###########################################################
        ''' fixed bug, occured when -10 not converted to tensor '''
        e = torch.exp(-10 * x)
        return (e - 1) / (torch.exp(torch.tensor(-10)) - 1)
        ###########################################################


class SqrtFunction(nn.Module):
    def forward(self, x):
        return torch.sqrt(x)


class SinhFunction(nn.Module):
    def forward(self, x):
        x = (torch.sinh(3*x)) / 10
        return x


class PowerFunction(nn.Module):
    def forward(self, x):
        x = (torch.pow(1000, x) - 1) / 1000.0
        return x


class SquaredFunction(nn.Module):
    def forward(self, x):
        x = x**2
        return x


class ASinhFunction(nn.Module):
    def forward(self, x):
        x = (torch.arcsinh(10*x)) / 3
        return x


class LinearFunction(nn.Module):
    def forward(self ,x):
        return x


class Stack(nn.Module):
    def __init__(self,):
        super().__init__()
        self.filters = [
            LinearFunction(),
            LogFunction(),
            PowerFunction(),
            SqrtFunction(),
            SquaredFunction(),
            ASinhFunction(),
            SinhFunction(),
        ]
        
        with open('./yoda/stats.yaml', 'r') as file:
            self.stats = yaml.safe_load(file)

    def scale_data(self, x, vmax=1.0):
        x = torch.clip(x, 0.0, 1.0)
        # Normalize
        x = torch.true_divide(x, vmax)
        x = torch.clip(x, 0.0, 1.0)
        return x

    def forward(self, x):
        x = torch.clip(x, 0.0, 1.0)
        
        x_list = [torch.clone(x) for _ in range(len(self.filters))]
        
        x_filtered = []
        for func, x_c in zip(self.filters, x_list):
            # Apply filter
            cur_x_filtered = func(x_c)
            
            # Scaling
            if isinstance(func, LinearFunction):
                vmax = self.stats['Linear']
                cur_x_filtered = self.scale_data(cur_x_filtered, vmax)
            elif isinstance(func, LogFunction):
                vmax = self.stats['Log']
                cur_x_filtered = self.scale_data(cur_x_filtered, vmax)
            elif isinstance(func, PowerFunction):
                vmax = self.stats['Power']
                cur_x_filtered = self.scale_data(cur_x_filtered, vmax)
            elif isinstance(func, SqrtFunction):
                vmax = self.stats['Sqrt']
                cur_x_filtered = self.scale_data(cur_x_filtered, vmax)
            elif isinstance(func, SquaredFunction):
                vmax = self.stats['Squared']
                cur_x_filtered = self.scale_data(cur_x_filtered, vmax)
            elif isinstance(func, ASinhFunction):
                vmax = self.stats['ASINH']
                cur_x_filtered = self.scale_data(cur_x_filtered, vmax)
            elif isinstance(func, SinhFunction):
                vmax = self.stats['SINH']
                cur_x_filtered = self.scale_data(cur_x_filtered, vmax)
            
            x_filtered.append(cur_x_filtered)
            # print(cur_x_filtered.size())
            # im_shape = 672
            # if cur_x_filtered.size(2) == im_shape and isinstance(func, LogFunction):
            #     import matplotlib.pyplot as plt
            #     im_to_plot = cur_x_filtered[0].detach().numpy().reshape((im_shape, im_shape))
            #     plt.imsave('inspect_yolomodelstack.png', im_to_plot, cmap='gray')
            #     # raise KeyboardInterrupt
        
        stacked_im = torch.cat(x_filtered, dim=1)
        return stacked_im

# Github References: https://github.com/MegEngine/RepLKNet/blob/main/model_replknet.py#L14
class ConvBn2d(nn.Module):
    def __init__(self, channels, kernel_size,):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=kernel_size//2, bias=False) # Keep same spatial res
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        return self.bn(self.conv(x))


class LargeKernel(nn.Module):
    def __init__(self, channels, kernel, small_kernels=()):
        super().__init__()
        self.dw_large = ConvBn2d(channels, kernel)

        self.small_kernels = small_kernels
        for k in self.small_kernels:
            setattr(self, f"dw_small_{k}", ConvBn2d(channels, k))

    def forward(self, inp):
        outp = self.dw_large(inp)
        for k in self.small_kernels:
            outp += getattr(self, f"dw_small_{k}")(inp)
        return outp


class ReduceDimBlock(nn.Module):
    # Reudce dimension by 1
    def __init__(self, channels, kernel, small_kernels=(), activation=nn.SiLU):
        super().__init__()
        # self.pw1 = nn.Conv2d(channels, channels, 1, 1, 0)
        # self.pw1_act = activation()
        self.lk = LargeKernel(channels, kernel, small_kernels)
        self.lk_act = activation()
        self.pw2 = nn.Conv2d(channels, channels-1, 1, 1, 0)
    
    def forward(self, x):
        # pw1_out = self.pw1_act(self.pw1(x))
        lk_out = self.lk_act(self.lk(x))
        pw2_out = self.pw2(lk_out)
        
        return pw2_out


class RouteBlock(nn.Module):
    def __init__(self, channels, kernel, small_kernels=(), activation=nn.SiLU) -> None:
        super().__init__()
        self.reduce1 = ReduceDimBlock(channels, kernel, small_kernels, activation) # 7 -> 6
        self.reduce2 = ReduceDimBlock(channels - 1, kernel, small_kernels, activation) # 6 -> 5
        self.reduce3 = ReduceDimBlock(channels - 2, kernel, small_kernels, activation) # 5 -> 4
        self.reduce4 = ReduceDimBlock(channels - 3, kernel, small_kernels, activation) # 4 -> 3
        self.reduce5 = ReduceDimBlock(channels - 4, kernel, small_kernels, activation) # 3 -> 2
        self.reduce6 = ReduceDimBlock(channels - 5, kernel, small_kernels, activation) # 2 -> 1
    
    def forward(self, x):
        x = self.reduce1(x)
        x = self.reduce2(x)
        x = self.reduce3(x)
        x = self.reduce4(x)
        x = self.reduce5(x)
        x = self.reduce6(x)
        return x


class CustomSqueeze(nn.Module):
    # CustomSqueeze from 7ch -> 1ch
    def __init__(self, inp_ch, kernel=1,) -> None:
        super().__init__()
        self.r1 = self.reducing_channel(inp_channel=inp_ch, kernel=kernel)
        self.r2 = self.reducing_channel(inp_channel=inp_ch - 1, kernel=kernel)
        self.r3 = self.reducing_channel(inp_channel=inp_ch - 2, kernel=kernel)
        self.r4 = self.reducing_channel(inp_channel=inp_ch - 3, kernel=kernel)
        self.r5 = self.reducing_channel(inp_channel=inp_ch - 4, kernel=kernel)
        self.r6 = self.reducing_channel(inp_channel=inp_ch - 5, kernel=kernel)
        
    
    def reducing_channel(self, inp_channel, kernel=1):
        return nn.Sequential(
            nn.Conv2d(inp_channel, inp_channel, kernel, 1, padding=kernel//2),
            nn.BatchNorm2d(inp_channel),
            nn.SiLU(),
            nn.Conv2d(inp_channel, inp_channel - 1, 1, 1, padding=0),   # 1x1
        )
    
    def forward(self, x):
        x = self.r1(x)
        x = self.r2(x)
        x = self.r3(x)
        x = self.r4(x)
        x = self.r5(x)
        x = self.r6(x)
        return x

class LargeKernelNet(nn.Module):
    def __init__(self, channels, kernels=(7, 5)) -> None:
        super().__init__()
        self.large_kernels = kernels
        for k in self.large_kernels:
            setattr(self, f'route_k{k}', CustomSqueeze(inp_ch=channels,
                                                       kernel=k))
        
        self.conv1x1 = nn.Conv2d(len(kernels), 1, 1, 1, 0)
    
    def forward(self, x):
        outputs = []
        for k in self.large_kernels:
            outp = getattr(self, f"route_k{k}")(x)
            outputs.append(outp)
        
        y = torch.cat(outputs, dim=1)
        y = self.conv1x1(y)
        
        return y


class YODA(nn.Module):
    def __init__(self, channels, kernels=(7, 5)) -> None:
        super().__init__()
        self.largeknet = LargeKernelNet(channels, kernels)
        self.stack = Stack()
    
    def forward(self, x):
        stacked_im = self.stack(x)
        latent_im = self.largeknet(stacked_im)
        return latent_im, stacked_im