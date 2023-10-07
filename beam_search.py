import re
import string
import contractions as con
from nltk.tokenize import TreebankWordTokenizer
import numpy as np


class BeamSearchDecoder:
    def __init__(self,model,x_vocab,y_vocab):
        """
        News summarizer class for summarizing news text to headlines
        :param model: tf transformer model class
        :param x_vocab: news text vocabulary
        :param y_vocab: news summary vocabulary
        """
        self.model = model
        self.xvoc = x_vocab
        self.yvoc = y_vocab
        self.maxlen = model.maxlen
        self.encoded_input = None

    def __call__(self,text):
        self.encoded_input = self.clean_text(text)
        self.encoded_input = self.xvoc.tokenize(self.encoded_input)
        self.encoded_input = self.xvoc.add_padding(self.encoded_input,self.maxlen)

    def summarize(self,k=1,alpha=1,ymax=25):
        """
        beam search decoder function
        :param k: beam size (k=1 is greedy search)
        :param alpha: sequence length normalization factor
        :param ymax: max output length
        :return: top 5 highest scoring summaries
        """
        # reshape input sequence to (1,maxlen)
        en_seq = self.encoded_input
        en_seq = np.asarray([en_seq], dtype=np.int32)

        # decoder input sequence starting with sos token
        dec_seq = np.asarray([[2]], dtype=np.int32)

        # get top k predicted tokens and scores
        pred = self.model([en_seq, dec_seq]).numpy()
        score = np.log(pred[:, -1:, :][0][0])
        topk_seq = np.argpartition(score, -k)[-k:]

        # create k beams
        beams = [([2] + [topk_seq[i]], score[topk_seq[i]]) for i in range(k)]
        ended_beams = []

        while len(beams) > 0:

            # predict new sequences and create new beams with top k beams

            new_beams = []

            for i, (seq, _) in enumerate(beams):
                dec_seq = np.asarray([seq], dtype=np.int32)
                pred = self.model([en_seq, dec_seq]).numpy()
                probs = np.log(pred[:, -1:, :][0][0])
                topk_seq = np.argpartition(probs, -k)[-k:]
                new_beam = [(seq + [x], beams[i][1] + probs[x]) for x in topk_seq]
                new_beams.extend(new_beam)

            new_beams = sorted(new_beams, key=lambda x: x[-1], reverse=True)[:k]

            beams = []

            # store ongoing beams and remove completed beams
            for seq, score in new_beams:
                if seq[-1] == 3 or len(seq) == ymax:
                    if len(ended_beams) < 5:
                        ended_beams.append((seq, score))
                        ended_beams = sorted(ended_beams, key=lambda x: x[-1], reverse=True)
                    else:
                        if min(ended_beams, key=lambda x: x[1])[1] < score:
                            ended_beams[-1] = (seq, score)

                else:
                    beams.append((seq, score))

        # normalize beam scores and convert to token idx to words
        ended_beams = [(x[0], x[1] / (len(x[0]) ** alpha)) for x in ended_beams]
        ended_beams = sorted(ended_beams, key=lambda x: x[1], reverse=True)
        ended_beams = [(' '.join([self.yvoc.i2w[t] for t in x[0]]), x[1]) for x in ended_beams]
        return ended_beams

    @staticmethod
    def clean_text(text, tk=TreebankWordTokenizer()):
        # fix contractions and remove URLs
        text = [con.fix(w) for w in text.split()]
        text = ' '.join(text).lower()
        url_pattern = r'http\S+|www\S+'
        text = re.sub(url_pattern, '', text)

        # remove punctuations other than , . : and provide space
        punct = string.punctuation.replace('.', '').replace(',', '').replace(':', '')
        translation_table = str.maketrans(punct, ' ' * len(punct))
        result_string = text.translate(translation_table)
        text = re.sub(r'\s+', ' ', result_string)
        text = tk.tokenize(text)
        text = ' '.join(text)
        text = re.sub(r'([!"#$%&\'()*+,-./:;<=>?@\\^_`{|}~])', r' \1 ', text)
        return text.replace(" s ", " 's ")

