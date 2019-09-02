from tensorboardX import SummaryWriter

class Logger:
    def __init__(self):
        print("Setting up TensorboardX")
        self.writer = SummaryWriter()

    def __del__(self):
        self.writer.close()

    def log_train_and_validation_accuracy(self, train_acc, val_acc, n_iter, rel):
        self.writer.add_scalars(rel + '/Accuracy', {'training': train_acc, 'validation': val_acc}, n_iter)

    def log_train_and_validation_ap(self, train_ap, val_ap, n_iter, rel):
        self.writer.add_scalars(rel + '/AP', {'training': train_ap, 'validation': val_ap}, n_iter)

    def log_loss(self, train_loss, test_loss, n_iter, rel):
        self.writer.add_scalars(rel + '/Loss', {'training': train_loss, "testing": test_loss}, n_iter)

    def log_accuracy(self, train_acc, val_acc, test_acc, n_iter, rel):
        self.writer.add_scalars(rel + '/Accuracy', {'training': train_acc, 'validation': val_acc, "testing": test_acc}, n_iter)

    def log_ap(self, train_ap, test_ap, n_iter, rel):
        self.writer.add_scalars(rel + '/AP', {'training': train_ap, "testing": test_ap}, n_iter)

    def log_param(self, name, param, n_iter):
        self.writer.add_histogram(name, param, n_iter)

    def close(self):
        self.writer.close()