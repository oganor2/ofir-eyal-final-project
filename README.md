# Final Project


**This repository contains the code for classifying users by demographics bla bla.** <br>

## Repository Structure

```
Final Project/
|-- src/
|   |-- main.py
|   |-- utils.py
|   |-- trainer.py
|   |-- models.py
|   |-- predict.py
|   |-- create_dataset.py
|
|-- data/
|   |-- tweets etc.
|
|-- config.yaml
|-- README.md
|-- requirements.txt
```
## Usage 

The code can be ran from a CLI using the following:<br>
```
 python src/main.py [-h] [-Action {train,evaluate,inference}] [-config CONFIG] [-download_data {y,yes,n,no}] [-book_data_path BOOK_DATA_PATH] [-experiment_name EXPERIMENT_NAME] 
 [-train_config TRAIN_CONFIG] [-model_path MODEL_PATH] [-text TEXT] [-output OUTPUT]
```
Arguments: <br>
```
  -h, --help: show this help message and exit 
  -Action, -a {train,evaluate,inference}:  The desired action. Note that train will run evaluation as well 
  -config :  configuration file *.y(a)ml 
  -download_data, -d {y,yes,n,no}:  Whether to download the dataset from gutenberg. Will default to yes if book_data_path is not provided and previous data not found 
  -book_data_path, -b: Path to location for saving/loading book data csv file
  -experiment_name, -e:  Name for saving resulting model + log, only relevant if train selected
  -train_config, -tc: Dictionary in the format {'batch_size':int,'num_epochs':int,'seq_len':int'}, only relevant if train selected. Default: 30,25,150 
  -model_path, -m: Path to model. Must be provided if evaluate or inference selected
  -text, -t:   Path to a text file for inference or evaluation. Note that evaluation will be meaningless for text that is not fully punctuated. Must be provided if inference selected 
  -output, -o: Path to output annotated text, if not provided will use stdout 
 ```
The trained model will be saved under a new directory named "models/", the includes all saved models, and a directory "models/logs/" that contains the relevant log for the experiment.
