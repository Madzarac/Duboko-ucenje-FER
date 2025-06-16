import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = img.view(img.size(0), -1)
        return feats

class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE
        self.append(nn.BatchNorm2d(num_features=num_maps_in))
        self.append(nn.ReLU(inplace=True))
        self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size=k, bias=bias))

class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.convnet = nn.Sequential(
            _BNReluConv(input_channels, emb_size, k=3),
            nn.MaxPool2d(kernel_size=3, stride=2),

            _BNReluConv(emb_size, emb_size, k=3),
            nn.MaxPool2d(kernel_size=3, stride=2),

            _BNReluConv(emb_size, emb_size, k=3),
        )

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = self.convnet(img)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1) 
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        margin = 1.0
        distance_positive = F.pairwise_distance(a_x, p_x, p=2)
        distance_negative = F.pairwise_distance(a_x, n_x, p=2)
        loss = torch.relu(distance_positive - distance_negative + margin).mean()
        return loss