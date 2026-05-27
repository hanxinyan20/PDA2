import torch.nn as nn
import torch
class RankNet(nn.Module):
    def __init__(self, num_features):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(num_features,
                                             512), nn.Dropout(0.2), nn.ReLU(),
                                   nn.Linear(512, 256), nn.Dropout(0.2),
                                   nn.ReLU(), nn.Linear(256, 128),
                                   nn.Dropout(0.2), nn.ReLU(),
                                   nn.Linear(128, 1))
        self.output = nn.Sigmoid()
    def forward(self, input1, input2):
        s1 = self.model(input1)
        s2 = self.model(input2)
        diff = s1 - s2
        prob = self.output(diff)
        return prob
    
class SetNN(nn.Module):
    def __init__(self, num_features, emb_size=128):
        super(SetNN, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(num_features, 128), nn.ReLU(),
                                     nn.Linear(128, emb_size), nn.Sigmoid())
        
        self.rank_net = RankNet(emb_size)

        self.ot_head = nn.Sequential(nn.Linear(emb_size, 128), nn.ReLU(),
                                     nn.Linear(128, 1))
        
    def forward(self, input1, input2):
        '''
        input shape (batch_size, set_size, num_features)
        '''
        emb_1 = self.encoder(input1).mean(dim=1) # (batch_size, set_size, emb)-> (batch_size, emb)
        emb_2 = self.encoder(input2).mean(dim=1) # (batch_size, set_size, emb)-> (batch_size, emb)
        prob = self.rank_net(emb_1 , emb_2)
        return self.ot_head(emb_1), self.ot_head(emb_2), prob
    
    def embed(self, input):
        '''
        input shape (set_size, num_features)
        
        return shape (emb_size)
        '''
        with torch.no_grad():
            return self.encoder(input).mean(dim=0)  
