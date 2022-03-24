import torch
import numpy as np
import pandas as pd
import transformers
from transformers import AutoModel, AutoTokenizer
import copy
import warnings
import os
from pprint import pprint

warnings.filterwarnings("ignore")
torch.random.manual_seed(42)
np.random.seed(42)
device = torch.cuda.is_available()

attributes = ['gender', 'political', 'children', 'race', 'income', 'education', 'age']


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, text_col='text_tokenized'):
        self.data = data
        self.text_col = text_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vals = []
        above, below = None, None
        row = self.data.iloc[idx]
        if idx:
            above = self.data.iloc[idx - 1]
            if above['user_id'] != row['user_id']:
                above = None
        if idx != self.__len__() - 1:
            below = self.data.iloc[idx + 1]
            if below['user_id'] != row['user_id']:
                below = None
        input_ids = copy.deepcopy(row[self.text_col]['input_ids'][:42])
        token_type_ids = copy.deepcopy(row[self.text_col]['token_type_ids'][:42])
        attention_mask = copy.deepcopy(row[self.text_col]['attention_mask'][:42])

        if above is not None:
            input_ids.extend(above[self.text_col]['input_ids'][:42])
            token_type_ids.extend(above[self.text_col]['token_type_ids'][:42])
            attention_mask.extend(above[self.text_col]['attention_mask'][:42])
        else:
            input_ids.extend(row[self.text_col]['input_ids'][:42])
            token_type_ids.extend(row[self.text_col]['token_type_ids'][:42])
            attention_mask.extend(row[self.text_col]['attention_mask'][:42])

        if below is not None:
            input_ids.extend(below[self.text_col]['input_ids'][:42])
            token_type_ids.extend(below[self.text_col]['token_type_ids'][:42])
            attention_mask.extend(below[self.text_col]['attention_mask'][:42])
        else:
            input_ids.extend(row[self.text_col]['input_ids'][:42])
            token_type_ids.extend(row[self.text_col]['token_type_ids'][:42])
            attention_mask.extend(row[self.text_col]['attention_mask'][:42])

        vals.append(input_ids)
        vals.append(token_type_ids)
        vals.append(attention_mask)
        return tuple([torch.tensor(i) for i in vals])


class ClassifierHead(torch.nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 100),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(100),
            torch.nn.Linear(100, output_dim),
            torch.nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class DemographicsModel(torch.nn.Module):
    def __init__(self, sample):
        """
        :param sample: a sample batch, to find bert output size
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
        embedding_dim = self.get_bert_output_size(sample)
        self.gender = ClassifierHead(embedding_dim, 2)
        self.political = ClassifierHead(embedding_dim, 3)
        self.has_children = ClassifierHead(embedding_dim, 2)
        self.race = ClassifierHead(embedding_dim, 3)
        self.income = ClassifierHead(embedding_dim, 2)
        self.education = ClassifierHead(embedding_dim, 2)
        self.age = ClassifierHead(embedding_dim, 2)

    def get_bert_output_size(self, sample):
        # to find bert output size
        return self.bert(**sample).pooler_output.shape[-1]

    def forward(self, tokenized):
        x = self.bert(**tokenized).pooler_output
        gender = self.gender(x)
        political = self.political(x)
        has_children = self.has_children(x)
        race = self.race(x)
        income = self.income(x)
        education = self.education(x)
        age = self.age(x)
        return gender, political, has_children, race, income, education, age


def get_pred_by_mode(categories, user_y_pred):
    example_pred = user_y_pred[0]
    mode_by_category = {cat: np.zeros(len(example_pred[cat])) for cat in categories}
    for pred in user_y_pred:
        for cat in pred:
            mode_by_category[cat][np.argmax(pred[cat])] += 1
    for i in categories:
        mode_by_category[i] /= len(user_y_pred)
    return mode_by_category


def get_pred_by_mean(categories, user_y_pred):
    example_pred = user_y_pred[0]
    mean_by_category = {cat: np.zeros(len(example_pred[cat])) for cat in categories}
    for pred in user_y_pred:
        for cat in pred:
            mean_by_category[cat] += pred[cat]
    for i in categories:
        mean_by_category[i] /= len(user_y_pred)
    return mean_by_category


class Predict:
    def __init__(self, model, data_set, attributes, batch_size=32, device=False, log=False):
        if type(model) == str:
            self.model = torch.load(model, map_location=torch.device('cpu') if not device else None)
        else:
            self.model = model
        self.test_dataloader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                                           num_workers=0)
        self.device = device
        self.attributes = attributes

    def inference_single_user(self):
        user_dict = {'y_pred': []}
        if self.device:
            self.model.cuda()
        self.model.eval()
        for batch in self.test_dataloader:
            if self.device:
                batch = tuple(t.cuda() for t in batch)
            input_ids, token_type_ids, input_mask = batch[:3]
            with torch.no_grad():
                outputs = self.model({'input_ids': input_ids, 'token_type_ids': token_type_ids,
                                      'attention_mask': input_mask})
            outputs = [o.detach().cpu().numpy() for o in outputs]
            for val in range(len(input_ids)):
                user_preds = [np.exp(o[val]) for o in outputs]
                user_dict['y_pred'].append({attribute: pred for attribute, pred in zip(self.attributes, user_preds)})
        return user_dict


def predict_one_user(user_tweets_df, ##less_labels
                     attributes=('gender', 'political', 'children', 'race', 'income', 'education', 'age'),
                     model_path=os.path.join("../models", 'BertFinetunedLessLabels'), mode_mean='mean', device=False):
    test_per_user_dataset = InferenceDataset(user_tweets_df)
    print(model_path)
    predict = Predict(model_path, test_per_user_dataset, attributes=attributes, device=False)
    predictions = predict.inference_single_user()
    if mode_mean == 'mean':
        return get_pred_by_mean(attributes, predictions['y_pred'])
    else:
        return get_pred_by_mode(attributes, predictions['y_pred'])

# def predict_one_user(user_tweets_df,
#                      attributes=('gender', 'political', 'children', 'race', 'income', 'education', 'age'),
#                      model_path=os.path.join("../models", 'BertFinetunedBinaryLabelsOnly'), mode_mean='mean', device=False):
#     test_per_user_dataset = InferenceDataset(user_tweets_df)
#     print(model_path)
#     predict = Predict(model_path, test_per_user_dataset, attributes=attributes, device=False)
#     predictions = predict.inference_single_user()
#     if mode_mean == 'mean':
#         return get_pred_by_mean(attributes, predictions['y_pred'])
#     else:
#         return get_pred_by_mode(attributes, predictions['y_pred'])


# def predict_one_user(user_tweets_df,
#                      attributes=('gender', 'political', 'children', 'race', 'income', 'education', 'age'),
#                      model_path=os.path.join("../models", 'BertFinetunedLessLabelsAugmented'), mode_mean='mean', device=False):
#     test_per_user_dataset = InferenceDataset(user_tweets_df)
#     print(model_path)
#     predict = Predict(model_path, test_per_user_dataset, attributes=attributes, device=False)
#     predictions = predict.inference_single_user()
#     if mode_mean == 'mean':
#         return get_pred_by_mean(attributes, predictions['y_pred'])
#     else:
#         return get_pred_by_mode(attributes, predictions['y_pred'])


