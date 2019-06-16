import torch
import torch.nn as nn

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1)

    def forward(self, x):
        out = self.net(x)
        return out

class TemporalDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalDeconvBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.deconv1, self.bn1, self.relu1, self.dropout1)
 
    def forward(self, x):
        out = self.net(x)
        return out

class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()
        # location_input ~ [batch_size, 2, 10]
        # Encoder
        self.l1_conv = TemporalConvBlock(in_channels=2, out_channels=32, kernel_size=3 , stride=1, dilation= 1,
                                     padding=0, dropout= 0.05) #-> [batch_size, 32, 6]
        # Decoder
        self.l1_deconv = TemporalDeconvBlock(in_channels= 32, out_channels= 32, kernel_size=3 , stride=1, dilation= 1,
                                    padding=0, dropout=  0.05) #-> [batch_size, 64, 4]

        # Last layer
        self.last = TemporalConvBlock(in_channels=32, out_channels=2, kernel_size=1 , stride=1, dilation= 1,
                                     padding=0, dropout=  0.05) #-> [batch_size, 32, 6]
        # combines all layers
        self.network = nn.Sequential(self.l1_conv,  self.l1_deconv, self.last)

        self.criter = torch.nn.MSELoss()

    def prepare_inputs(self, inputs, mean, std):

        pos_x, pos_y = inputs[:2]
        # Locations
        # Note: prediction target is displacement from last input
        x = (pos_x - mean)/std
        y = (pos_y - mean)/std
        return x, y
    
    def mse(self, data, target):
        temp = (data - target)**2
        rmse = torch.mean(temp[:,0,:] + temp[:,1,:])
        return rmse

    def forward(self, inputs):
        mean = torch.tensor([640., 476.23620605])
        std = torch.tensor([227.59802246, 65.00177002])
        pos_x, pos_y = self.prepare_inputs(inputs, mean, std)
        pos_x = pos_x.permute(0,2,1)   
        pos_y = pos_y.permute(0,2,1)   
        pred_y = self.network(pos_x)

        #swap back
        loss = self.criter(pred_y, pos_y)
        #pred_y = pred_y.permute(0,2,1)   
        #pred_y = pred_y*std + mean 

        return loss, pred_y