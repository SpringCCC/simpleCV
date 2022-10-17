from torch import nn

class CBR(nn.Module):


    def __init__(self, in_c, out_c, k=3, s=1, p=1, norm="bn", relu="leakrelu"):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        if norm=="bn":
            self.norm = nn.BatchNorm2d(out_c)
        else:
            self.norm = nn.InstanceNorm2d(out_c)
        if relu=='relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))