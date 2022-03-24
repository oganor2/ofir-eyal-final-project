from utils import *


class PunctuationModel(torch.nn.Module):
    def __init__(self,num_classes, sample):
        """
        :param num_classes: number of punctuation types
        :param sample: a sample batch, to find bert output size
        """
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        embedding_dim = self.get_bert_output_size(sample)
        self.before = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 100),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(100),
            torch.nn.Linear(100, num_classes),
            torch.nn.LogSoftmax(dim=-1)
        )
        self.after = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 100),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(100),
            torch.nn.Linear(100, num_classes),
            torch.nn.LogSoftmax(dim=-1)
        )
        self.capital = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 100),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(100),
            torch.nn.Linear(100, 2),
            torch.nn.LogSoftmax(dim=-1)
        )
        self.br = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 100),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(100),
            torch.nn.Linear(100, 2),
            torch.nn.LogSoftmax(dim=-1)
        )

    def get_bert_output_size(self, sample):
        # to find bert output size
        return self.bert(**sample).last_hidden_state.shape[-1]

    def forward(self, tokenized):
        x = self.bert(**tokenized).last_hidden_state
        before = self.before(x)
        after = self.after(x)
        capital = self.capital(x)
        br = self.br(x)
        return before, after, capital, br
