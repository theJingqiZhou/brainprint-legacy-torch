import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


class Extractor_log_spec(nn.Module):
    def __init__(
        self,
        n_mels=12,
        sample_rate=1000,
        n_fft=100,
        hop_length=10,
        window="hann",
        window_size=100,
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    ):
        super(Extractor_log_spec, self).__init__()

        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            freeze_parameters=freeze_parameters,
        )

    def forward(self, input):
        channel_num = input.shape[1]
        feats = []
        for ch_id in range(channel_num):
            x = self.spectrogram_extractor(input[:, ch_id, :])
            x = self.logmel_extractor(x)
            feats.append(x)

        feat = torch.concat(feats, dim=1)
        return feat


class ResNet50(nn.Module):
    def __init__(
        self,
    ):
        super(ResNet50, self).__init__()
        self.extractor_log_spec = Extractor_log_spec(
            n_mels=12, sample_rate=1000, n_fft=100, hop_length=10, window_size=100
        )

        resnet = models.resnet50(pretrained=True)
        self.pretrained = nn.Module()
        self.pretrained.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )
        self.pretrained.layer2 = resnet.layer2
        self.pretrained.layer3 = resnet.layer3
        self.pretrained.layer4 = resnet.layer4
        self.pretrained.pool = resnet.avgpool

        self.feature_dim = 2048

    def forward(self, x):
        x = self.extractor_log_spec(x)
        for i in range(x.shape[1]):
            if i == 0:
                x_ = x[:, i, :, :]
            else:
                x_ = torch.cat((x_, x[:, i, :, :]), dim=2)
        x = torch.unsqueeze(x_, dim=1)
        x = F.interpolate(x, size=(224, 224))
        x = x.repeat(1, 3, 1, 1)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        feat = self.pretrained.pool(layer_4)
        feat = feat.squeeze(-1)
        return feat.squeeze(-1)
