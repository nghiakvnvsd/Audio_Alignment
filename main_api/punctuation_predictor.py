import torch
import os
import string
import re
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from torch import nn
from torchcrf import CRF
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoConfig, RobertaForTokenClassification)

import random


device = torch.device("cpu")
OVERLAP_LENGTH = 20
MAX_LENGTH = 200


class PuncRobertaLstmCrfModel(RobertaForTokenClassification):
    def __init__(self, config):
        super(PuncRobertaLstmCrfModel, self).__init__(config=config)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size // 2, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, device='cuda'):
        sequence_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)[0]
        sequence_output, _ = self.lstm(sequence_output)

        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]

        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=attention_mask_label.type(torch.uint8))
            return -1.0 * log_likelihood
        else:
            sequence_tags = self.crf.decode(logits)
            return sequence_tags


class InputExample():
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures():
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


class PunctuationPredictor:
    def __init__(self, model_dir, max_seq_length=MAX_LENGTH, overlap_length=OVERLAP_LENGTH, batch_size=8):
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.label_list = ['O', 'PERIOD', 'COMMA', 'COLON', 'QMARK', 'EXCLAM', 'SEMICOLON', self.tokenizer.cls_token, self.tokenizer.sep_token]
        self.label_map = {i: label for i, label in enumerate(self.label_list, 1)}
        self.config = AutoConfig.from_pretrained(os.path.join(model_dir, "config"))
        self.model = PuncRobertaLstmCrfModel.from_pretrained(os.path.join(model_dir, "model"),
                                                        from_tf=False,
                                                        config=self.config)
        # model.resize_token_embeddings(len(tokenizer))
        self.model.to(device)
        self.dot_list = list("!.;?|")

        self.max_seq_length = max_seq_length
        self.overlap_length = overlap_length
        self.batch_size = batch_size

        self.punc_map = {
            "PERIOD": ".",
            "COMMA": ",",
            "COLON": ":",
            "QMARK": "?",
            "EXCLAM": "!",
            "SEMICOLON": ";",
        }
        assert max_seq_length > overlap_length, "Max sequence length must be greater than overlap length"

    def get_examples_from_sentences(self, sentences):
        """See base class."""
        paragraphs = []
        raw_paragraphs = []
        token_labels = []
        overlap = []
        half_overlap_length = self.overlap_length // 2
        remain_overlap_length = self.overlap_length - half_overlap_length
        for word_list in sentences:
            idx = 0
            n_tokens = len(word_list)
            while idx < n_tokens:
                if idx > 0:
                    idx -= self.overlap_length
                    overlap[-half_overlap_length:] = [True] * half_overlap_length

                end_idx = min(idx + self.max_seq_length, n_tokens)
                words = word_list[idx: end_idx]
                labels = ["O"] * len(words)
                new_words = []
                for i, w in enumerate(words):
                    if len(w) > 0 and w.strip()[-1] == ",":
                        new_words.append(w.replace(",", ""))
                        labels[i] = "COMMA"
                raw_paragraphs.append(words)
                paragraphs.append([w.lower() for w in words])
                token_labels.append(labels)

                if idx > 0:
                    overlap.extend([True] * remain_overlap_length + [False] * (len(words) - remain_overlap_length))
                else:
                    overlap.extend([False] * len(words))
                idx = end_idx
            if len(token_labels) > 0 and len(token_labels[-1]) > 0:
                token_labels[-1][-1] = "PERIOD"

        result = list(zip(paragraphs, token_labels))
        return self._create_examples(result, "infer"), raw_paragraphs, overlap

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def convert_examples_to_features(self, examples, noise_prob = 0.3, mode = 'eval', add_noise=True):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = {label : i for i, label in enumerate(self.label_list, 1)}

        features = []
        loop_times = [0, 1] if mode == 'train' else [0]
        for (ex_index,example) in enumerate(examples):
          for t in loop_times:
            textlist = example.text_a.split(' ')
            labellist = example.label
            tokens = []
            labels = []
            valid = []
            label_mask = []
            num_to_noise = noise_prob * len(textlist)
            count_noise = 0
            for i, word in enumerate(textlist):
                if add_noise and t == 1:
                  if random.random() < noise_prob and count_noise < num_to_noise:
                    word = remove_accents(word)
                    count_noise += 1
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
            if len(tokens) >= self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]
                labels = labels[0:(self.max_seq_length - 2)]
                valid = valid[0:(self.max_seq_length - 2)]
                label_mask = label_mask[0:(self.max_seq_length - 2)]
            ntokens = []
            label_ids = []
            ntokens.append(self.tokenizer.cls_token)
            valid.insert(0, 1)
            label_mask.insert(0, 1)
            label_ids.append(label_map[self.tokenizer.cls_token])
            for i, token in enumerate(tokens):
                ntokens.append(token)
                if len(labels) > i:
                    label_ids.append(label_map[labels[i]])
            ntokens.append(self.tokenizer.sep_token)
            valid.append(1)
            label_mask.append(1)
            label_ids.append(label_map[self.tokenizer.sep_token])
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                label_ids.append(0)
                valid.append(1)
                label_mask.append(0)
            while len(label_ids) < self.max_seq_length:
                label_ids.append(0)
                label_mask.append(0)
            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length
            assert len(valid) == self.max_seq_length
            assert len(label_mask) == self.max_seq_length

            features.append(InputFeatures(input_ids=input_ids,
                                          input_mask=input_mask,
                                          label_id=label_ids,
                                          valid_ids=valid,
                                          label_mask=label_mask))
        return features

    def predict(self, text):
        # text = text.lower()

        # convert other punctuation to dot
        for p in self.dot_list:
            text = text.replace(p, ".")
        text = text.replace("\n", " ")

        # remove duplicate spaces
        text = re.sub(' +', ' ', text)

        sentences = text.strip().split(". ")
        sentences = [line.strip().split(" ") for line in sentences]
        eval_examples, paragraphs, overlap = self.get_examples_from_sentences(sentences)
        word_list = []
        for line in paragraphs:
            word_list.extend(line)
        print("Word list", word_list)
        eval_features = self.convert_examples_to_features(eval_examples)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_valid_ids,
                                  all_lmask_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)
        self.model.eval()
        y_pred = []
        y_true = []
        for input_ids, input_mask, label_ids, valid_ids, l_mask in tqdm(eval_dataloader, desc="Predicting"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = self.model(input_ids, None, input_mask, valid_ids=valid_ids,
                                    attention_mask_label=l_mask, device="cpu")

            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(self.label_map):
                        y_true.extend(temp_1)
                        y_pred.extend(temp_2)
                        break
                    else:
                        temp_1.append(self.label_map[label_ids[i][j]])
                        temp_2.append(self.label_map.get(logits[i][j], 'PAD'))

        result = ""
        prev_is_eos = False
        first_word = True
        for i, label in enumerate(y_true):
            if label != "O":
                y_pred[i] = label

        for word, label, skip in zip(word_list, y_pred, overlap):
            if skip:
                continue

            if label == "O":
                result += word.title() if prev_is_eos or first_word else word
                result += " "
                prev_is_eos = False
            else:
                for _, p in self.punc_map.items():
                    word = word.rstrip(p)
                result += word + self.punc_map.get(label, "") + " "
                prev_is_eos = label != "COMMA"

            if first_word:
                first_word = False
        result = result.strip()
        print("Result", result)
        return result


if __name__ == "__main__":
    text = """
    Giáo viên dạy Anh nói chuyện với một giáo viên khác :”Tui không thể chịu nổi sao lại có đứa học trò thế này.Chuyện là tôi có ra một bài làm là hãy kể một câu chuyện ngắn bằng tiếng Anh , rồi nó kể câu chuyện về hoàng tử và công chúa”.
    Giáo viên kia thắc mắc:
    _ Vậy có gì không ổn?
    _ Không ổn là bài làm của nó như thế này:”Hoàng tử và công chúa gặp nhau tại lâu đài. Hoàng tử hỏi :””Can you speak Vietnamese?”” Công chúa trả lời:””Sure””.Thế là sau đó cả bài văn nó toàn viết bằng tiếng Việt hết.
    """

    print("Removing all punctuation marks...")
    punc_list = string.punctuation + "…“”‘’"
    text = text.lower().translate(str.maketrans(punc_list, ' '*len(punc_list)))

    model = PunctuationPredictor("./weights/checkpoint_30501/")
    result = model.predict(text)

    print("Original:\n", text)
    print("====================================================")
    print("Result:\n", result)
