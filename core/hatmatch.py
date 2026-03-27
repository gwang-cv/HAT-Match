import torch
import torch.nn as nn
from loss import batch_episym
from einops import rearrange
import torch.nn.functional as F


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel)
        )

    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)
        out = out + x1
        return torch.relu(out)


class MaxDGCNN_Block(nn.Module):
    def __init__(self, knn_num=9, in_channel=128):
        super(MaxDGCNN_Block, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel


        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel*2, self.in_channel, (1, 1)), 
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)), 
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            )


    def forward(self, x):
        B, _, N, _ = x.shape 
        out = self.conv(x)  
        out = out.max(dim=-1, keepdim=False)[0]
        out = out.unsqueeze(3)
        return out
     

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)   
    xx = torch.sum(x ** 2, dim=1, keepdim=True)   
    pairwise_distance = -xx - inner - xx.transpose(2, 1)   

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   

    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None): 
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)   
    if idx is None:
        idx_out = knn(x, k=k)    
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx_out + idx_base  

    idx = idx.view(-1)   

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)   
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)   
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()   
    return feature


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LCT(nn.Module):
    def __init__(self, channels, num_heads, k_num=20):
        super(LCT, self).__init__()
        self.k_num = k_num
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))   

        self.query_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        self.key_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        self.value_filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU())
        self.gcn_q = MaxDGCNN_Block(knn_num=self.k_num, in_channel=channels)
        self.gcn_k = MaxDGCNN_Block(knn_num=self.k_num, in_channel=channels)
        self.gcn_v = MaxDGCNN_Block(knn_num=self.k_num, in_channel=channels)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=(1, 1))

    def forward(self, x):
        B, C, N, _ = x.shape
        q = self.query_filter(x)
        k = self.key_filter(x)
        v = self.value_filter(x)

        q = self.gcn_q(get_graph_feature(q, k=self.k_num))
        k = self.gcn_k(get_graph_feature(k, k=self.k_num))
        v = self.gcn_v(get_graph_feature(v, k=self.k_num))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=N, w=1)

        out = self.project_out(out)
        return out + x


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
        return out


class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))

    def forward(self, x_up, x_down):
        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out

# global graph block 
class GG_Block(nn.Module):
    def __init__(self, in_channel):
        super(GG_Block, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def soft_adjacent_matrix(self, w):
        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm(w, w.transpose(1, 2)) 
        return A

    def sgda(self, x, w):
        B, _, N, _ = x.size() 
        with torch.no_grad():
            A_S = self.soft_adjacent_matrix(w) 
            A_S = torch.softmax(A_S,dim=-1)
            I_N = torch.eye(N).unsqueeze(0).to(x.device).detach() 
            A_S2 = A_S + I_N 
            D_out = torch.sum(A_S2, dim=-1) 
            D_S = torch.diag_embed(D_out)
        out = x.squeeze(-1).transpose(1, 2).contiguous() 
        out = torch.bmm(D_S, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous() 

        return out

    def forward(self, x, w):
        out = self.sgda(x, w)
        out = self.conv(out)
        return out
 
class LCTM(nn.Module):
    def __init__(self, channels, num_heads=4, k_num=8):
        super(LCTM, self).__init__()
        self.k_num = k_num 
        self.attn = LCT(channels, num_heads, self.k_num) 
        self.ffn = ResNet_Block(channels, channels, pre=False)

    def forward(self, x):
        b, c, h, w = x.shape 
        x = x + self.attn(x) 
        x = x + self.ffn(x)

        return x  


class SEAttention(nn.Module):
    def __init__(self, channel, reduction):
        nn.Module.__init__(self)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.ffn = ResNet_Block(channel, channel, pre=False)
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = x * y.expand_as(x)
        out = out + self.ffn(out)
        return out
 

class CL_Block(nn.Module):
    def __init__(self, initial=False, predict=False, out_channel=128, k_num=8, sampling_rate=0.5, clusters=200):
        super(CL_Block, self).__init__()
        self.initial = initial 
        self.in_channel = 6 if self.initial is True else 8   
        self.out_channel = out_channel
        self.k_num = k_num
        self.predict = predict
        self.sr = sampling_rate
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),   
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.down_1 = diff_pool(self.out_channel, clusters)
        self.l1 = []
        for _ in range(2):
            self.l1.append(OAFilter(self.out_channel, clusters))
        self.up_1 = diff_unpool(self.out_channel, clusters)
        self.l1 = nn.Sequential(*self.l1)

        self.down_2 = diff_pool(self.out_channel, clusters)
        self.l2 = []
        for _ in range(2):
            self.l2.append(OAFilter(self.out_channel, clusters))
        self.up_2 = diff_unpool(self.out_channel, clusters)
        self.l2 = nn.Sequential(*self.l2)
 

        self.embed_00 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )
        self.embed_002 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
            ResNet_Block(self.out_channel, self.out_channel, pre=False),
        )

        self.gg_01 = GG_Block(self.out_channel) 

        self.resfomer_1 = LCTM(self.out_channel, num_heads=4, k_num=self.k_num)
        self.resfomer_2 = LCTM(self.out_channel, num_heads=4, k_num=self.k_num)

        self.seattn_1 = SEAttention(self.out_channel,reduction=2)
        self.seattn_2 = SEAttention(self.out_channel,reduction=2)

        self.resnet_1 = ResNet_Block(self.out_channel * 2, self.out_channel, pre=True)
        self.resnet_2 = ResNet_Block(self.out_channel * 2, self.out_channel, pre=True)

        self.embed_1 = nn.Sequential(
            ResNet_Block(self.out_channel, self.out_channel, pre=False)
        ) 

        self.linear_0 = nn.Conv2d(self.out_channel, 1, (1, 1))
        self.linear_1 = nn.Conv2d(self.out_channel, 1, (1, 1)) 

        if self.predict:
            self.embed_2 = ResNet_Block(self.out_channel, self.out_channel, pre=False)
            self.linear_2 = nn.Conv2d(self.out_channel, 2, (1, 1))

    def down_sampling(self, x, y, weights, indices, features=None, predict=False):
        B, _, N , _ = x.size()
        indices = indices[:, :int(N*self.sr)]   
        with torch.no_grad():
            y_out = torch.gather(y, dim=-1, index=indices)   
            w_out = torch.gather(weights, dim=-1, index=indices) 
        indices = indices.view(B, 1, -1, 1)   

        if predict == False:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))   
            return x_out, y_out, w_out
        else:
            with torch.no_grad():
                x_out = torch.gather(x[:, :, :, :4], dim=2, index=indices.repeat(1, 1, 1, 4))   
            feature_out = torch.gather(features, dim=2, index=indices.repeat(1, 128, 1, 1))   
            return x_out, y_out, w_out, feature_out

    def forward(self, x, y):
        B, _, N, _ = x.size()
        out = x.transpose(1, 3).contiguous()   
        out = self.conv(out)   

### layer1
        x_down = self.down_1(out)
        x2 = self.l1(x_down)
        x_up = self.up_1(out, x2)


        out = self.embed_00(x_up)   

        x1 = self.resfomer_1(out)

        x1s = self.seattn_1(out)

        out = self.resnet_1(torch.cat([x1, x1s], dim=1))

### layer2

        x_down = self.down_2(out)
        x2 = self.l2(x_down)
        x_up = self.up_2(out, x2)


        out = self.embed_002(x_up)   

        x1 = self.resfomer_2(out)

        x1s = self.seattn_2(out)

        out = self.resnet_2(torch.cat([x1, x1s], dim=1))



        w0 = self.linear_0(out).view(B, -1)    


        out = out + self.gg_01(out, w0.detach()) 

        out = self.embed_1(out)
        w1 = self.linear_1(out).view(B, -1)   


        if self.predict == False:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds = self.down_sampling(x, y, w0, indices, None, self.predict)
            return x_ds, y_ds, [w0, w1], [w0_ds, w1_ds]
        else:
            w1_ds, indices = torch.sort(w1, dim=-1, descending=True)
            w1_ds = w1_ds[:, :int(N * self.sr)]
            x_ds, y_ds, w0_ds, out = self.down_sampling(x, y, w0, indices, out, self.predict)
            out = self.embed_2(out)
            w2 = self.linear_2(out)
            e_hat = weighted_8points(x_ds, w2)

            return x_ds, y_ds, [w0, w1, w2[:, 0, :, 0]], [w0_ds, w1_ds], e_hat


class HATNet(nn.Module):
    def __init__(self, config):
        super(HATNet, self).__init__()

        self.ds_0 = CL_Block(initial=True, predict=False, out_channel=128, k_num=9,
                             sampling_rate=config.sr, clusters=config.clusters)   
        self.ds_1 = CL_Block(initial=False, predict=True, out_channel=128, k_num=6, sampling_rate=config.sr, clusters=config.clusters)

    def forward(self, x, y):
        x0 = x 
        B, _, N, _ = x.shape
        xx1 = x[:,:,:,:2]
        xx2 = x[:,:,:,2:]
        motion = xx1 - xx2
        x = torch.cat([x, motion], dim=-1)

        x1, y1, ws0, w_ds0 = self.ds_0(x, y)   
        B1, _, N1, _ = x1.shape
        xx1s = x1[:,:,:,:2]
        xx2s = x1[:,:,:,2:]
        motions = xx1s - xx2s
        x1 = torch.cat([x1, motions], dim=-1)


        w_ds0[0] = torch.relu(torch.tanh(w_ds0[0])).reshape(B, 1, -1, 1)   
        w_ds0[1] = torch.relu(torch.tanh(w_ds0[1])).reshape(B, 1, -1, 1)
        x_ = torch.cat([x1, w_ds0[0].detach(), w_ds0[1].detach()], dim=-1)   

        x2, y2, ws1, w_ds1, e_hat = self.ds_1(x_, y1)   

        with torch.no_grad():
            y_hat = batch_episym(x0[:, 0, :, :2], x0[:, 0, :, 2:], e_hat)   

        return ws0 + ws1, [y, y, y1, y1, y2], [e_hat], y_hat


def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)
    for batch_idx in range(X.shape[0]):
        e, v = torch.symeig(X[batch_idx, :, :].squeeze(), True)
        bv[batch_idx, :, :] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    mask = logits[:, 0, :, 0]
    weights = logits[:, 1, :, 0]

    mask = torch.sigmoid(mask)
    weights = torch.exp(weights) * mask
    weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-5)

    x_shp = x_in.shape
    x_in = x_in.squeeze(1)

    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1).contiguous()

    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1).contiguous()
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1).contiguous(), wX)

    # Recover essential matrix from self-adjoing eigen

    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat




