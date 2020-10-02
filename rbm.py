import torch
from torchvision import datasets, transforms

class RBM:
    def __init__(self, visible_units, hidden_units, lr=0.001, momentum=0.5, weight_decay=0.0001):
        self.hidden_bias = torch.zeros(hidden_units)
        self.visible_bias = torch.ones(visible_units)*0.5
        self.weights = torch.randn(visible_units, hidden_units) * 0.1

        self.momentum = momentum
        self.lr = lr
        self.weight_decay = weight_decay

        self.weights_velocity = torch.zeros(visible_units, hidden_units)
        self.visible_bias_velocity = torch.zeros(visible_units)
        self.hidden_bias_velocity = torch.zeros(hidden_units)

    def generate_activations(self, prob):
        #random_mtx = torch.rand(prob.shape)
        #activations = (prob > random_mtx).float()
        return torch.bernoulli(prob)



    def visible_to_hidden_prob(self, visible_prob):
        hidden_state = visible_prob @ self.weights + self.hidden_bias
        hidden_prob = torch.sigmoid(hidden_state)
        #print('visible_to_hidden: hidden_prob.shape = {}'.format(hidden_prob.shape))
        return hidden_prob, self.generate_activations(hidden_prob)

    def hidden_to_visible_prob(self, hidden_prob):
        visible_state = hidden_prob @ self.weights.T + self.visible_bias
        visible_prob = torch.sigmoid(visible_state)
        #print('visible_to_hidden: visible_prob.shape = {}
        return visible_prob, self.generate_activations(visible_prob)


    def delta_weight(self,  visible_act_data, hidden_act_data, visible_act_model, hidden_act_model):
        dw = (hidden_act_data.T @ visible_act_data) - (hidden_act_model.T @ visible_act_model)
        #print('delta_weight: delta_weight.shape = {}'.format(dw.shape))
        return dw.T

    def delta_bias(self, data_bias, model_bias):
        db = torch.sum(data_bias - model_bias, dim=0)
        return db


    def train_step(self, input_sample):
        hidden_prob_data, hidden_act_data = self.visible_to_hidden_prob(input_sample)
        hidden_act_model = hidden_act_data.clone()
        for ix in range(2):
            visible_prob_model, visible_act_model = self.hidden_to_visible_prob(hidden_act_model)
            hidden_prob_model, hidden_act_model = self.visible_to_hidden_prob(visible_prob_model)

        delta_w = self.delta_weight(input_sample, hidden_act_data, visible_prob_model, hidden_prob_model )
        delta_v = self.delta_bias(input_sample, visible_prob_model)
        delta_h = self.delta_bias(hidden_prob_data, hidden_prob_model)
        loss = torch.nn.functional.mse_loss(input_sample, visible_prob_model)

        return delta_w, delta_v, delta_h, loss

    def update_parameters(self, delta_w, delta_v, delta_h, batch_size):
        self.weights_velocity = self.momentum*self.weights_velocity + delta_w
        self.visible_bias_velocity = self.momentum*self.visible_bias_velocity + delta_v
        self.hidden_bias_velocity = self.momentum*self.hidden_bias_velocity + delta_h
        self.weights += self.weights_velocity * self.lr / batch_size
        self.visible_bias += self.visible_bias_velocity * self.lr / batch_size
        self.hidden_bias += self.hidden_bias_velocity * self.lr / batch_size
        self.weights = self.weights * (1-self.weight_decay)

    def train(self, train_loader, device, num_epochs=10):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = torch.reshape(data, (data.shape[0], -1))
                delta_w, delta_v, delta_h, loss = self.train_step(data)
                batch_size = data.shape[0]
                self.update_parameters(delta_w, delta_v, delta_h, batch_size)
                total_loss = total_loss + loss
                avg_loss = total_loss/(batch_idx+1)
                print('Epoch ={}, Batch Idx ={}, Loss = {}'.format(epoch, batch_idx, avg_loss))
            print('Epoch ={}, Loss = {}'.format(epoch,total_loss))


batch_size=64
test_batch_size=64
if torch.cuda.is_available():
    print('Cuda is available')
else:
    print('Cuda is not available')

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('Cuda is used')
else:
    print('Cuda is not used')

torch.manual_seed(0)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           #transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)


rbm = RBM(784, 256)
rbm.train(train_loader, device)