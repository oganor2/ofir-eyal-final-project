from utils import *
import copy


def f1_per_label(labels, output, num_classes):
    """
    simple utility to get f1-score per label, can be nan
    """
    scores = np.zeros(num_classes)
    for punc in range(num_classes):
        cur_labels, cur_output = (labels == punc)[labels != -1], (output == punc)[labels != -1]
        if not sum(cur_labels):
            scores[punc] = np.nan
        else:
            scores[punc] = f1_score(cur_labels, cur_output, zero_division=0)
    return scores


def ids_to_words(ids, predictions, tokenizer, return_words=False):
    """
    Convert embedded words, predictions back to words/chars. Also adjusts the annotations to match
    the true length of the words (since the tokenizer may change it)
    :param ids: embedded words
    :param predictions: embedded predictions
    :param tokenizer: the tokenizer used for embedding
    :param return_words: whether to return words or only labels
    """
    words = [tokenizer.convert_ids_to_tokens(i) for i in ids]
    all_tokens, all_labels = [], []
    for tokens, prediction in zip(words, predictions):
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, prediction):
            # can ignore the rest
            if token == '[PAD]':
                break
            # means that tokenizer split the word
            if token.startswith("##"):
                # recombine
                new_tokens[-1] = new_tokens[-1] + token[2:]
                # try and get a better label for the previous, rather than just discarding the label
                if new_labels[-1] == 11:
                    new_labels[-1] = label_idx
                elif not new_labels[-1] and label_idx == 1:
                    new_labels[-1] = label_idx
            else:
                new_labels.append(label_idx)
                new_tokens.append(token)
        all_tokens.append(new_tokens), all_labels.append(new_labels)
    if return_words:
        return all_tokens
    else:
        return all_labels


class Predict:
    def __init__(self, model, data_set, batch_size=16, device=False,log=False):
        if type(model) == str:
            self.model = torch.load(model, map_location=torch.device('cpu') if not device else None)
        else:
            self.model = model
        self.test_dataloader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False,
                                                           num_workers=0)
        self.device = device
        # only used if ran after training
        self.log = log
    def _step(self, batch):
        if self.device:
            batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, before_l, after_l, capital_l, br_l = batch
        labels = [before_l, after_l, capital_l, br_l]
        with torch.no_grad():
            outputs = self.model({'input_ids': input_ids, 'attention_mask': input_mask})
        labels = [l.cpu().numpy() for l in labels]
        outputs = [o.detach().cpu().numpy() for o in outputs]
        return input_ids, labels, outputs

    def evaluate(self):
        self.model.eval()
        accuracy, f1, = [], []
        print('Evaluating on data:')
        for batch in tqdm(self.test_dataloader):
            input_ids, labels, outputs = self._step(batch)
            # Calculate the accuracy for this batch of test sentences.
            cur_accuracy, cur_f1 = [], []
            zipped_o_l = [i for i in zip(outputs, labels)]
            for vals in range(len(zipped_o_l)):
                o, l = zipped_o_l[vals]
                # convert to account for added token
                o = np.array([j for i in ids_to_words(input_ids.cpu().numpy(),
                                                      np.argmax(o, -1), tokenizer) for j in i])
                l = np.array([j for i in ids_to_words(input_ids.cpu().numpy(),
                                                      l, tokenizer) for j in i])
                cur_accuracy.append((np.array(o == l).sum(-1) / l.shape).mean())
                if vals > 1:
                    cur_f1.append(f1_per_label(o, l, 2))
                else:
                    cur_f1.append(f1_per_label(o, l, 12))
            accuracy.append(cur_accuracy)
            f1.append(cur_f1)
        print("#############################################################")
        print(f"Test Accuracy: {np.round(np.nanmean(accuracy, 0), 4)}")
        print(f"Test f1-score for (before, after) the following classes\n"
              f"{punc} is:\n"
              f"{[i for i in zip(punc, np.round(np.nanmean([i[0] for i in f1], 0), 4))]}\n"
              f"{[i for i in zip(punc, np.round(np.nanmean([i[1] for i in f1], 0), 4))]}\n"
              f"for capitals :     {np.round(np.nanmean([i[2] for i in f1], 0), 4)}\n"
              f"for line breaks:   {np.round(np.nanmean([i[3] for i in f1], 0), 4)}")
        if self.log:
            logging.info("#############################################################")
            logging.info(f"Test Accuracy: {np.round(np.nanmean(accuracy, 0), 4)}")
            logging.info(f"Test f1-score for (before, after) the following classes\n"
                  f"{punc} is:\n"
                  f"{[i for i in zip(punc, np.round(np.nanmean([i[0] for i in f1], 0), 4))]}\n"
                  f"{[i for i in zip(punc, np.round(np.nanmean([i[1] for i in f1], 0), 4))]}\n"
                  f"for capitals :     {np.round(np.nanmean([i[2] for i in f1], 0), 4)}\n"
                  f"for line breaks:   {np.round(np.nanmean([i[3] for i in f1], 0), 4)}")

    def annotate(self):
        annotation = {'before': '', 'after': '', 'capital': '', 'break': ''}
        full_text, annotations = [], []
        self.model.eval()
        for step, batch in enumerate(self.test_dataloader):
            batch_annotation = [copy.deepcopy(annotation) for i in
                                range(batch[0].shape[0] * batch[0].shape[1])]
            # ignore labels
            input_ids, _, outputs = self._step(batch)
            for output_type in range(len(outputs)):
                # convert to account for added token
                o = outputs[output_type]
                predictions = np.array([j for i in ids_to_words(input_ids.cpu().numpy(),
                                                                np.argmax(o, -1), tokenizer) for j in i])
                for i in range(len(predictions)):
                    batch_annotation[i][list(annotation)[output_type]] = predictions[i]
            text = np.array([j for i in ids_to_words(input_ids.cpu().numpy(),
                                                     np.argmax(o, -1), tokenizer, return_words=True) for j
                             in i])
            full_text.extend(text)
            annotations.extend([i for i in batch_annotation if i != annotation])
        return full_text, annotations

    def result(self, output_path=None):
        no_space_after = ['"', "'", '(', '-']
        no_space_before = ['!', ',', '-', '.', ':', ';', '?', "'", ')']
        full_text, annotations = self.annotate()
        annotated = [i for i in zip(full_text, annotations)]
        result = []
        n_words = len(annotated)
        prev_add_break = ""
        for i in range(n_words):
            word, annotation = annotated[i]
            if annotation['capital']:
                word = word.capitalize()
            annotation = translate_annotation(annotation)

            if not i:
                result.append(word)
            else:
                prev_annotation = translate_annotation(annotated[i - 1][1])
                best_pred = get_optimal_annotation(prev_annotation['after'], annotation['before'])
                result.extend([best_pred[0], prev_add_break, best_pred[1], word])
            if i == n_words - 1:
                result.append(annotation['after'])
            prev_add_break = "<br>" if annotation['break'] else ""
        n_words = len(result)
        for i in range(n_words):
            word = result[i]
            if not word or word == 'None':
                continue
            if word == '<br>':
                word = "\n\n"
            if i + 1 < n_words - 1:
                add_space = " " if (word not in no_space_after and result[i + 1] not in no_space_before) \
                    else ""
            else:
                add_space = ""
            if output_path is not None:
                with open(output_path, 'w') as f:
                    print(word, file=f, end=add_space)
            else:
                print(word, end=add_space)
        if result[-1] not in ['.', '<br>']:
            print('.')
        print('\n')
        
