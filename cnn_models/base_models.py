import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cifar10_module


class ExampleNet(nn.Module):
    def __init__(self, args, input_shape):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, sample_dict):
        x = sample_dict["image"].cuda()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return {"prediction": x}


class ExampleNet_generic(nn.Module):
    def _flatten_feat_size(self, shape):
        o = self.features(torch.zeros(1, *shape).cuda())
        return int(np.prod(o.size()))

    def _feat_shape(self, shape):
        o = self.features(torch.zeros(1, *shape).cuda())
        return o.size()

    def freeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def add_convolution_layer(self, input_shape, nfilters, kernel_size):
        # features = nn.Sequential()

        cur_shape = self._feat_shape(input_shape)
        extra_conv = nn.Conv2d(
            cur_shape[1],
            out_channels=nfilters,
            kernel_size=kernel_size,
            padding=int(kernel_size / 2),
        )

        self.features.add_module(repr(len(self.features)), extra_conv.cuda())
        self.features.add_module(repr(len(self.features)), nn.ReLU())
        # self.features.add_module(repr(len(self.features)), extra_conv.cuda())
        # self.features.add_module(repr(len(self.features)), nn.ReLU())

        feat_lenght = self._flatten_feat_size(input_shape)
        self.classifier = nn.Sequential(
            nn.Linear(feat_lenght, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        ).cuda()

    def __init__(self, args, input_shape):
        super(ExampleNet_generic, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 6, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.features = self.features.cuda()
        conv_out_size = self._flatten_feat_size(input_shape)
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
        self.classifier.to("cuda:0")

    def forward(self, sample_dict):
        x = sample_dict["image"].cuda()
        conv_out = self.features(x).view(x.size()[0], -1)
        return {"prediction": self.classifier(conv_out)}


class TinyCifar(nn.Module):
    def _flatten_feat_size(self, shape):
        o = self.features(torch.zeros(1, *shape).cuda())
        return int(np.prod(o.size()))

    def _feat_shape(self, shape):
        o = self.features(torch.zeros(1, *shape).cuda())
        return o.size()

    def freeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def __init__(self, args, input_shape):
        super(TinyCifar, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.features = self.features.cuda()
        self.conv_out_size = self._flatten_feat_size(input_shape)
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_out_size, 64),
            nn.Linear(64, 10),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.classifier.to("cuda:0")

    def forward(self, sample_dict):
        x = sample_dict["image"].cuda()
        conv_out = self.features(x).view(x.size()[0], -1)
        return {"prediction": self.classifier(conv_out)}


class cifar10quick(nn.Module):
    def _flatten_feat_size(self, shape):
        o = self.features(torch.zeros(1, *shape).cuda())
        return int(np.prod(o.size()))

    def _feat_shape(self, shape):
        o = self.features(torch.zeros(1, *shape).cuda())
        return o.size()

    def freeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def __init__(self, args, input_shape):
        super(cifar10quick, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.features = self.features.cuda()
        conv_out_size = self._flatten_feat_size(input_shape)
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_size, 64),
            nn.Linear(64, args.n_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.classifier.to("cuda:0")

    def forward(self, sample_dict):
        x = sample_dict["image"].cuda()
        conv_out = self.features(x).view(x.size()[0], -1)
        return {"prediction": self.classifier(conv_out)}


class TorchVisionWrapper(torch.nn.Module):
    def __init__(self, args, input_shape=None):
        super(TorchVisionWrapper, self).__init__()
        self.net = cifar10_module.get_classifier(
            args, args.torch_model, pretrained=False
        )
        # self.net = torch.hub.load('pytorch/vision:v0.5.0', args.torch_model, pretrained=False)
        # self.net.fc = nn.Linear(512, args.n_classes)
        self.net = self.net.cuda()

    def forward(self, input_dict):
        out = self.net(input_dict["image"])
        return {"prediction": out}


class TorchVisionFeaturesWrapper(torch.nn.Module):
    def __init__(self, args, input_shape=None):
        super(TorchVisionWrapper, self).__init__()
        self.net = cifar10_module.get_classifier(
            args, args.torch_model, pretrained=False, features=True
        )
        # self.net = torch.hub.load('pytorch/vision:v0.5.0', args.torch_model, pretrained=False)
        # self.net.fc = nn.Linear(512, args.n_classes)
        self.net = self.net.cuda()

    def forward(self, input_dict):
        out = self.net(input_dict["image"])
        return {"prediction": out}


class decision_maker_logit(torch.nn.Module):
    def __init__(self, args, input_shape=None):
        super(decision_maker_logit, self).__init__()
        self.args = args
        input_size = len(args.cifar_classifier_indexes) * 2 * args.n_classes
        output_size = args.n_classes
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_size),
        )
        self.fc = self.fc.cuda()

    def forward(self, input_dict):
        logits = self.fc(input_dict["image"])
        # log_prob = F.log_softmax(logits, dim=1)
        return {"prediction": logits}


class decision_maker_deep(torch.nn.Module):
    def __init__(self, args, input_shape=None):
        super(decision_maker_deep, self).__init__()
        self.args = args
        input_size = len(args.cifar_classifier_indexes) * 2 * args.n_classes
        output_size = args.n_classes
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )
        self.fc = self.fc.cuda()

    def forward(self, input_dict):
        logits = self.fc(input_dict["image"])
        # log_prob = F.log_softmax(logits, dim=1)
        return {"prediction": logits}
