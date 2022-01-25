import torch
from torch import nn
from torch import optim
from torch.nn.utils import prune

from predictive_hands.training_methods.TrainingMethod import TrainingMethod


def arg_loss(predicted, target, soft_argmax):
    predicted_timing = soft_argmax(predicted)
    target_timing = torch.argmax(target,dim=1).float()
    return torch.nn.functional.mse_loss(predicted_timing,target_timing)

def kl_loss(predicted, target, logsoftmax, kl_div_loss):
    target_test = target / torch.sum(target)
    return kl_div_loss(logsoftmax(predicted), target_test)

def sliding_batch_norm(x):
    x_new = torch.zeros(x.shape)
    for i in range(0, x.shape[0]):
        x_seq = x[0:i]
        x_new[i] = (x[i] - torch.mean(x_seq))/(torch.sqrt(torch.var(x_seq) + .00001))
    return x_new


contact_loss_dict = {'kl_div': None, 'mse': nn.MSELoss(), 'arg_dist': None}

class DenseNetLSTM(TrainingMethod):

    class softArgmax(nn.Module):
        def __init__(self, cfg, device_type,beta=20):
            super().__init__()
            self.beta = beta
            self.device_type = device_type

        def forward(self, x):
            top = torch.exp(self.beta*x)
            bottom = torch.sum(top,dim=1)
            indices = torch.arange(top.shape[1]).to(self.device_type).float()
            return torch.einsum('j,ijk->ik', indices, top/bottom)


    class denseBlock(nn.Module):
        def __init__(self, cfg, device_type, input_dim, hidden_dim):
            super().__init__()
            self.device_type = device_type
            self.cfg = cfg

            self.norm1 = nn.BatchNorm1d(input_dim, track_running_stats=False)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu1 = nn.ReLU(hidden_dim)
            self.norm2 = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.relu2 = nn.ReLU(hidden_dim)

        def sliding_batch_norm(self, x):
            x_new = torch.zeros(x.shape, device=self.device_type)
            x_new[0] = x[0]
            for i in range(1, x.shape[0]):
                x_seq = x[0:i]
                x_new[i] = (x[i] - torch.mean(x_seq)) / (torch.sqrt(torch.var(x_seq) + .00001))
            return x_new

        def forward(self, x, batch=None):
            x = self.fc1(x.clone())
            x = self.relu1(x)

            x = self.fc2(x)
            x = self.relu2(x)

            return x

    class denseNetLSTMNetwork(nn.Module):
        def __init__(self, cfg, input_dim, predicted_dim, device_type, use_sigmoid):
            super().__init__()
            self.cfg = cfg
            self.device_type = device_type
            self.dense1 = DenseNetLSTM.denseBlock(cfg, device_type, input_dim, cfg['dense_dim_1'])
            self.dense2 = DenseNetLSTM.denseBlock(cfg, device_type,  input_dim + cfg['dense_dim_1'], cfg['dense_dim_2'])

            self.attn = nn.MultiheadAttention(input_dim + cfg['dense_dim_1'] + cfg['dense_dim_2'], num_heads=1)
            self.fc_attn = nn.Linear(input_dim + cfg['dense_dim_1'] + cfg['dense_dim_2'], predicted_dim)

            self.use_sigmoid = use_sigmoid

            if self.use_sigmoid:
                self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x1 = self.dense1(x)
            x2 = self.dense2(torch.cat((x,x1), dim=2))
            x = torch.cat((x,x1,x2),dim=2)
            x, _ = self.attn(x, x, x, attn_mask=torch.triu(torch.ones((x.shape[0], x.shape[0]), dtype=torch.bool),
                                                                diagonal=1).to(self.device_type))
            self.fc_attn(x)
            return x



    def __init__(self, cfg, input_dict, target_dict, device_type):
        self.cfg = cfg

        self.device_type = device_type

        self.loss_function = self.onset_loss

        in_data, target_data = self.dict_to_input(input_dict, target_dict, get_dims = True, to_cuda=False)

        input_dim = in_data.shape[2]
        predicted_dim = target_data.shape[2]

        use_sigmoid = False
        if self.cfg['contact_loss'] == 'arg_dist':
            self.contact_loss = lambda predicted, target: arg_loss(predicted, target, DenseNetLSTM.softArgmax())
            use_sigmoid = True
        elif self.cfg['contact_loss'] == 'kl_div':
            self.contact_loss = lambda predicted, target: kl_loss(predicted, target, nn.LogSoftmax(dim=1), nn.KLDivLoss(reduction='batchmean'))
        else:
            self.contact_loss = contact_loss_dict[self.cfg['contact_loss']]

        self.model = self.denseNetLSTMNetwork(self.cfg, input_dim, predicted_dim, self.device_type, use_sigmoid=use_sigmoid).to(device_type)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.cfg['lr']), weight_decay=float(self.cfg['decay']))
        self.loss = None
        self.recon_loss = nn.MSELoss()


    def onset_loss(self, predicted, target, alpha=0, beta=1):
        r_loss = self.recon_loss(predicted[:, :, self.recon_dims], target[:, :, self.recon_dims])
        if predicted.isnan().any() or target.isnan().any():
            print("hi")
        c_loss = self.contact_loss(predicted[:, :, self.contact_dims], target[:, :, self.contact_dims])
        r_loss = 0 if torch.isnan(r_loss) else r_loss
        c_loss = 0 if torch.isnan(c_loss) else c_loss
        return r_loss * alpha + c_loss * beta

    def train_epoch(self, input_batch, target_batch):
        torch.autograd.set_detect_anomaly(True)
        input_batch, target_batch = self.dict_to_input(input_batch, target_batch)
        lstm_in = input_batch.permute([1,0,2])
        self.model.zero_grad()
        predicted = self.model(lstm_in)
        predicted = predicted.permute([1,0,2])
        self.loss = self.loss_function(predicted, target_batch, alpha=self.cfg['train_objective'][0],
                                  beta=self.cfg['train_objective'][1])
        with torch.no_grad():
            train_loss = self.loss
        self.loss.backward()
        self.optimizer.step()
        return train_loss

    def val_epoch(self, input_dict, target_dict):
        input, target = self.dict_to_input(input_dict, target_dict)
        with torch.no_grad():
            lstm_in = input.permute([1,0,2])
            predicted = self.model(lstm_in)
            predicted = predicted.permute([1,0,2])
            val_loss = self.loss_function(predicted, target, alpha=self.cfg['train_objective'][0],
                                          beta=self.cfg['train_objective'][1])
            predicted_dict = self.prediction_to_dict(predicted, target_dict)
        return val_loss, predicted, predicted_dict

    def run_inference(self, input):
        with torch.no_grad():
            lstm_in = input.permute([1, 0, 2])
            predicted = self.model(lstm_in)
            predicted = predicted.permute([1, 0, 2]).cpu()
        return predicted

    def dict_to_input(self, input_dict, target_dict, get_dims = False, to_cuda=True):
        in_verts = torch.cat(list(torch.cat((input_dict['hand_points'][cur_hand], input_dict['dists'][cur_hand]), 3) for cur_hand in self.cfg['hands']), 2)

        in_verts = in_verts.view((in_verts.shape[0], in_verts.shape[1],-1))

        in_data = in_verts

        target_verts = torch.cat(list(target_dict['hand_points'][cur_hand]
                                      for cur_hand in self.cfg['hands']), 2)
        target_verts = target_verts.view((target_verts.shape[0], target_verts.shape[1], -1))
        target_data = torch.cat((target_verts, torch.cat([target_dict['contacts'][cur_hand] for cur_hand in self.cfg['hands']], 2)), 2)
        if get_dims:
            self.recon_dims = list(range(0, target_verts.shape[2]))
            self.contact_dims = list(range(target_verts.shape[2],target_data.shape[2]))

        if to_cuda:
            in_data = in_data.float().to(self.device_type)
            target_data = target_data.float().to(self.device_type)

        return in_data, target_data

    def prediction_to_dict(self, predicted_data, target_dict):
        predicted_dict = {}
        predicted_dict['hand_points'] = {}
        predicted_dict['contacts'] = {}
        prev_index = 0
        for hand in self.cfg['hands']:
            target_shape = target_dict['hand_points'][hand].shape
            predicted_dict['hand_points'][hand] = predicted_data[:, :, prev_index:prev_index + target_shape[2] * target_shape[3]]
            predicted_dict['hand_points'][hand] = predicted_dict['hand_points'][hand].view(target_shape)
            prev_index += target_shape[2] * target_shape[3]
        for hand in self.cfg['hands']:
            target_shape = target_dict['contacts'][hand].shape
            predicted_dict['contacts'][hand] = predicted_data[:, :, prev_index:prev_index + target_shape[2]]
            predicted_dict['contacts'][hand] = predicted_dict['contacts'][hand].view(target_shape)
            prev_index += target_shape[2]

        return predicted_dict




