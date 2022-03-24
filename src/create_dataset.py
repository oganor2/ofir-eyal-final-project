from utils import *
from langdetect import detect
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers


def download_data(meta_path='../data/gutenberg_metadata.csv',
                  output_path='../data/books_data.csv',save=True):
    df_metadata = pd.read_csv(meta_path)
    df_fiction = df_metadata[
        df_metadata.Bookshelf.isin(['Adventure', 'Crime', 'Detective', 'Fantasy', 'Mystery', 'Romantic'])]
    all_books = []
    for key, row in [i for i in df_fiction.iterrows()]:
        data = {'Author': None, 'Title': None, 'Link': None, 'ID': None, 'Bookshelf': None, 'Text': None}
        if data['Author'] is None:
            data['Author'] = [row['Author']]
        else:
            data['Author'].append(row['Author'])

        if data['Title'] is None:
            data['Title'] = [row['Title']]
        else:
            data['Title'].append(row['Title'])

        if data['Link'] is None:
            data['Link'] = [row['Link']]
        else:
            data['Link'].append(row['Link'])

        book_id = int(row['Link'].split('/')[-1])

        if data['ID'] is None:
            data['ID'] = [book_id]
        else:
            data['ID'].append(book_id)

        if data['Bookshelf'] is None:
            data['Bookshelf'] = [row['Bookshelf']]
        else:
            data['Bookshelf'].append(row['Bookshelf'])

        text = np.nan
        try:
            text = strip_headers(load_etext(etextno=book_id)).strip()
        except:
            print("Couldn't acquire text for " + row['Title'] + ' with ID ' +
                  str(book_id) + '. Link: ' + row['Link'])

        data['Text'] = text
        if data['Text'] is not None:
            all_books.append(data)

    df_data = pd.DataFrame(all_books, columns=['Title', 'Author', 'Link', 'ID', 'Bookshelf', 'Text'])
    df_data.dropna(inplace=True)
    print(f'Downloaded {df_data.shape[0]} books from Gutenberg project')
    if save:
        df_data.to_csv(output_path, index=False)
    return df_data


def create_gutenberg_dataset(df_path='../data/books_data.csv',download=False, save_df=True, seq_len=150,
                             save_dataset=None):
    if download:
        book_data = download_data(output_path=df_path, save=save_df)
    else:
        try:
            train, val, test = torch.load('../data/train_dataset',map_location=torch.device('cpu') if not device else None),\
                               torch.load('../data/val_dataset',map_location=torch.device('cpu') if not device else None),\
                               torch.load('../data/test_dataset',map_location=torch.device('cpu') if not device else None)
            print('Loaded existing datasets')
            return train, val,test
        except:
            try:
                book_data = pd.read_csv(df_path)
            except:
                book_data = download_data(output_path=df_path,save=save_df)

    book_data['lang'] = book_data.Text.apply(lambda x: detect(x))
    book_data = book_data[book_data.lang == 'en']
    book_data['Text'] = book_data.Text.apply(lambda x: clean_text(x))
    book_data['annotations'] = book_data.Text.apply(lambda x: annotate_text(x))
    book_data['raw_text'] = book_data.Text.apply(lambda x: to_raw(x))
    print('Creating datasets for model:')
    data = book_data.progress_apply(lambda x: prepare_for_batches(x['raw_text'], x['annotations'],
                                    tokenizer, seq_len, annotations_embedding), axis=1)
    train,val,test = data_to_dataset(data,seq_len)
    if save_dataset is not None:
        torch.save(train,'../data/train_dataset')
        torch.save(val,'../data/val_dataset')
        torch.save(test, '../data/test_dataset')
    return train, val, test


def create_prediction_dataset(text,raw=False, seq_len=150):
    text = clean_text(text, n=0)
    annotations = annotate_text(text,raw)
    raw_text = to_raw(text)
    data = prepare_for_batches(raw_text, annotations, tokenizer,seq_len,
                               annotations_embedding, test_percent=1)
    data_set = data_to_dataset([data], seq_len, train_val=False)
    return data_set



