from utils import *
from create_dataset import *
from trainer import Trainer
from predict import Predict
from models import PunctuationModel
import argparse
import yaml


"""
Default args for running
"""
configs = {'batch_size':30,'num_epochs':25,'seq_len':150}
meta_path = None
download_data = False

parser = argparse.ArgumentParser(description='Train a new  model on the original dataset or evaluate a '
                                             'saved model. Can also be used for inference on a text'
                                             ' file.')
parser.add_argument('-Action','-a',choices=['train','evaluate','inference'],default='train',
                    help='Desired action. Note that train will run evaluation as well')
parser.add_argument('-config', help="configuration file *.y(a)ml", default=None)
parser.add_argument('-download_data','-d',choices=['y','yes','n','no'], default='no',
                    help='Whether to download the dataset from gutenberg. Will default to yes'
                         'if book_data_path is not provided and previous data not found')
parser.add_argument('-book_data_path','-b',help='Path to location for saving/loading book data csv file.')
parser.add_argument('-experiment_name','-e',help='Name for saving resulting model + log, '
                    'only relevant if train selected', default='fine_tuned_bert')
parser.add_argument('-train_config','-tc',help="Dictionary in the format {'batch_size':int,"
                                               "'num_epochs':int,'seq_len':int'}, only relevant if train "
                                        "selected. Default: 30,25,150",default=configs)
parser.add_argument('-model_path','-m',help='Path to model. Must be provided if evaluate or '
                                            'inference selected')
parser.add_argument('-text','-t',help='Path to a text file for inference or evaluation. Note that '
                                      'evaluation will be meaningless for text that is not fully '
                                      'punctuated. Must be provided if inference selected')
parser.add_argument('-output','-o',help='Path to output annotated text, if not provided will use stdout')


if __name__ == "__main__":
    try:
        args = parser.parse_args()
        if args.config:
            opt = vars(args)
            args = yaml.load(open(args.config), Loader=yaml.FullLoader)
            opt.update(args)
            args = opt
        else:
            args = vars(args)
    except Exception as e:
        print(e)
        print('Please make sure yaml file is in the correct format or that arguments were provided')
        raise Exception


    if args['Action'] == 'train':
        train,val,test = create_gutenberg_dataset(args['book_data_path'],download=args['download_data'],
                                                  save_df=True,seq_len=args['train_config']['seq_len'])
        model = PunctuationModel
        trainer = Trainer(PunctuationModel,train,val,test,args['experiment_name'],**args['train_config'],
                          device=device)
        trainer.train()
        predictor = Predict(trainer.model,test,device=device,log=True)
        predictor.evaluate()

    elif args['Action'] == 'evaluate':
        model = torch.load(args['model_path'],map_location=torch.device('cpu') if not device else None)
        if args['text'] is not None:
            with open(args['text'], 'r') as f:
                text = f.readlines()
            text = " ".join(text)
            test = create_prediction_dataset(text, seq_len=args['train_config']['seq_len'])
        else:
            _,_,test = create_gutenberg_dataset(args['book_data_path'],download=args['download_data'],
                                                  save_df=False,seq_len=args['train_config']['seq_len'])
        predictor = Predict(model, test, device=device,log=True)
        predictor.evaluate()

    else:
        model = torch.load(args['model_path'],map_location=torch.device('cpu') if not device else None)
        try:
            with open(args['text'],'r') as f:
                text = f.readlines()
        except Exception as e:
            print(e)
            print('Please make sure you provided a valid text file')
            raise Exception
        text = " ".join(text)
        test = create_prediction_dataset(text,raw=True,seq_len=args['train_config']['seq_len'])
        predictor = Predict(model, test, device=device)
        predictor.result(args['output'])


