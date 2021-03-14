from torch.utils.tensorboard import SummaryWriter
import numpy as np

class MyLogger:
    def __init__(self, trainloader, model, suffix=''):
        self.writer = SummaryWriter('runs/experiment'+suffix)
        #images, labels = next(iter(trainloader))
        #grid = torchvision.utils.make_grid(images)
        #self.writer.add_image('images', grid, 0)
        #self.writer.add_graph(model, images)

    def __del__(self):
        self.writer.flush()
        self.writer.close()

    def log_train_loss(self, acc, loss, step):
        self.writer.add_scalar('Loss/train', loss, step)
        self.writer.add_scalar('Accuracy/train', acc, step)

    def log_train_running_loss(self, loss, step):
        self.writer.add_scalar('RunningLoss/train', loss, step)

    def log_test_loss(self, acc, loss, step):
        self.writer.add_scalar('Loss/test', loss, step)
        self.writer.add_scalar('Accuracy/test', acc, step)

    def log_test_running_loss(self, loss, step):
        self.writer.add_scalar('RunningLoss/test', loss,  step)

    def log_ae_running_loss(self, loss, step, tag):
        self.writer.add_scalar('RunningLoss/'+tag, loss, step)

    def log_ae_running_mse_loss(self, loss, step, tag):
        self.writer.add_scalar('RunningMSELoss/'+tag, loss, step)

    def log_eigvals(self, eigvals, layer_id, step):
        logval=np.ma.log(eigvals)
        logdet = np.sum(logval.filled(0))
        trace = np.sum(eigvals)
        #eigratio = np.max(eigvals)/np.percentile(eigvals,10)
        self.writer.add_scalar('Logdet/Layer' + str(layer_id), logdet, step)
        self.writer.add_scalar('Trace/Layer' + str(layer_id), trace, step)
        #self.writer.add_scalar('EigRatio/Layer' + str(layer_id), eigratio, step)
