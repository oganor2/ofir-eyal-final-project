from utils import *

from torch.optim import Adam
from torch.nn import NLLLoss
from torch.optim.lr_scheduler import ExponentialLR


class Trainer:
    def __init__(self, model_class, train, val, test, experiment_name, loss_fn=NLLLoss,
                 optimizer=Adam,batch_size=16, num_epochs=25, seq_len=150, device=False):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seq_len = seq_len
        self.device = device
        self.train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                                            num_workers=0)
        self.val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True,
                                                          num_workers=0)
        self.test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                                           num_workers=0)
        sample = next(iter(self.train_dataloader))
        self.model = model_class(len(annotations_embedding),
                                 {'input_ids': sample[0], 'attention_mask': sample[1]})
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.experiment_name = experiment_name
        if not os.path.exists("../models"):
            os.mkdir("../models")
        if not os.path.exists("../models/logs"):
            os.mkdir("../models/logs")
        self.model_path = os.path.join("../models",self.experiment_name)
        self.logging_path = os.path.join("../models/logs",f'{experiment_name}.log')
        log = logging.getLogger()
        for hdlr in log.handlers[:]:
            log.removeHandler(hdlr)
        logging.basicConfig(filename=self.logging_path, level=logging.INFO,
                            format='%(asctime)s - %(message)s')
        print(self.model)

    def train(self,scheduler=ExponentialLR, scheduler_steps=15,patience=10,print_step=400):
        parameters1 = self.model.bert.parameters()
        parameters2 = [*self.model.before.parameters(), *self.model.after.parameters(), *self.model.capital.parameters(),
                       *self.model.br.parameters()]
        optimizer1 = self.optimizer(parameters1, lr=3e-5)
        optimizer2 = self.optimizer(parameters2)
        if scheduler:
            scheduler1 = scheduler(optimizer1, 0.999)
            scheduler2 = scheduler(optimizer2, 0.999)
        loss_fn = self.loss_fn(ignore_index=-1)
        if self.device:
            self.model.cuda()

        loss_values, validation_loss_values = [], []
        min_loss = np.inf
        counter = 0

        for epoch in range(self.num_epochs):
            # ========================================
            # train

            self.model.train()
            total_loss = 0

            for step, batch in enumerate(self.train_dataloader):
                if self.device:
                    batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, before_l, after_l, capital_l, br_l = batch
                self.model.zero_grad()
                before_p, after_p, capital_p, br_p = self.model({'input_ids': input_ids,
                                                                 'attention_mask': input_mask})
                # need to swap the axes since using on a sequence, may raise issues if loss_fn isn't NLL
                loss_be, loss_af, loss_cap, loss_br = loss_fn(before_p.swapaxes(1, -1), before_l),\
                    loss_fn(after_p.swapaxes(1, -1), after_l),\
                    loss_fn(capital_p.swapaxes(1, -1), capital_l),\
                    loss_fn(br_p.swapaxes(1, -1), br_l)
                loss = loss_be + loss_af + loss_cap + loss_br
                loss.backward()
                total_loss += loss.item()

                optimizer1.step()
                optimizer2.step()

                if scheduler:
                    if not step % scheduler_steps:
                        scheduler1.step()
                        scheduler2.step()

                if not step % print_step:
                    print("#############################################################")
                    print(f"Current step is {step}/{len(self.train_dataloader)} Current batch loss is:"
                          f" {np.round(loss.item(), 4)}")
                    logging.info("#############################################################")
                    logging.info(f"Current step is {step}/{len(self.train_dataloader)} "
                                 f"Current batch loss is: {np.round(loss.item(), 4)}")
            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / (step + 1)
            print(f"Average train loss: {np.round(avg_train_loss, 4)} \n")
            logging.info(f"Average train loss: {np.round(avg_train_loss, 4)}")
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            # ========================================
            # validation

            self.model.eval()
            eval_loss = 0
            accuracy, f1, = [], []
            for step, batch in enumerate(self.val_dataloader):
                if self.device:
                    batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, before_l, after_l, capital_l, br_l = batch
                labels = [before_l, after_l, capital_l, br_l]
                with torch.no_grad():
                    outputs = self.model({'input_ids': input_ids, 'attention_mask': input_mask})
                for o, l in zip(outputs, labels):
                    eval_loss += loss_fn(o.swapaxes(1, -1), l).item()

                labels = [l.cpu().numpy() for l in labels]
                outputs = [o.detach().cpu().numpy() for o in outputs]

                # Calculate the accuracy and f1 for this batch of test sentences.
                cur_accuracy, cur_f1 = [], []
                for o, l in zip(outputs, labels):
                    cur_accuracy.append(np.array((np.argmax(o[l != -1], -1) == l[l != -1]).sum(-1) /
                                                 (l[l != -1]).shape).mean())
                    cur_f1.append(f1_score(np.argmax(o[l != -1], -1), l[l != -1], average='macro'))
                accuracy.append(cur_accuracy)
                f1.append(cur_f1)
            print("#############################################################")
            logging.info("#############################################################")
            avg_eval_loss = eval_loss / (step + 1)
            validation_loss_values.append(avg_eval_loss)
            if avg_eval_loss < min_loss:
                torch.save(self.model.state_dict(), self.model_path)
                print(f'Saved model ({self.experiment_name})')
                logging.info(f'Saved model ({self.experiment_name})')
                min_loss = avg_eval_loss
                counter = 0
            else:
                counter += 1
                if counter > patience:
                    break

            print(f"Epoch {epoch + 1}/{self.num_epochs} resulted in:")
            print(f"Validation loss: {np.round(avg_eval_loss, 4)}")
            print(f"Validation Accuracy: {np.round(np.array(accuracy).mean(0), 4)}")
            print(f"Validation f1-score: {np.round(np.array(f1).mean(0), 4)}")
            print('\n')
            logging.info(f"Epoch {epoch + 1}/{self.num_epochs} resulted in:")
            logging.info(f"Validation loss: {np.round(avg_eval_loss, 4)}")
            logging.info(f"Validation Accuracy: {np.round(np.array(accuracy).mean(0), 4)}")
            logging.info(f"Validation f1-score: {np.round(np.array(f1).mean(0), 4)}")
        return loss_values,validation_loss_values
