"""
RF Convolutional Decoder model for RF-based heart rate estimation
"""

from typing import Tuple
import torch


class RF_conv_encoder(torch.nn.Module):
    """
    RF Convolutional Encoder for processing RF signal data.
    
    This encoder processes RF (Radio Frequency) signals to extract features
    for heart rate estimation.
    """
    
    def __init__(self, channels=10):  
        super(RF_conv_encoder, self).__init__()
        
        self.ConvBlock1 = torch.nn.Sequential(
            torch.nn.Conv1d(channels, 32, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
        )

        self.ConvBlock2 = torch.nn.Sequential(
            torch.nn.Conv1d(32, 64, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
        )

        self.convBlock5 = torch.nn.Sequential(
            torch.nn.Conv1d(256, 512, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
        )

        self.ConvBlock6_mean = torch.nn.Sequential(
            torch.nn.Conv1d(512, 1024, 7, stride=1, padding=3),
        )

        
    def forward(self, x_orig: torch.tensor) -> torch.tensor:
        x = self.ConvBlock1(x_orig)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.convBlock5(x)
        x_encoded_mean  = self.ConvBlock6_mean(x)

        return x_encoded_mean


class RF_conv_decoder(torch.nn.Module):
    """
    RF Convolutional Decoder for heart rate estimation from RF signals.
    
    This model combines an encoder and decoder to process RF signals
    and output heart rate estimates.
    """
    
    def __init__(self, channels=10):  
        super(RF_conv_decoder, self).__init__()

        self.IQ_encoder = RF_conv_encoder(channels)

        self.ConvBlock1 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(1024, 512, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(512, 256, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(256, 128, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(128, 64, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(64, 32, 7, stride=1, padding=3),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(32, 1, 1, stride=1, padding=0)
        )
        
    def forward(self, x_IQ: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        # z_IQ : (B, 1024, T)
        z_IQ = self.IQ_encoder(x_IQ)
        x = self.ConvBlock1(z_IQ)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)

        x_feature = x  # (B, 128, T)

        x = self.ConvBlock4(x)   # (B, 64, T)
        x = self.ConvBlock5(x)   # (B, 32, T)
        x_decoded = self.ConvBlock6(x)  # (B, 1, T)
        x_decoded = x_decoded.squeeze(1)
        return x_decoded, x_feature
