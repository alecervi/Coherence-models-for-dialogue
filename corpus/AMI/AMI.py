from __future__ import print_function
from __future__ import unicode_literals
import os
from collections import OrderedDict
from corpus.Corpus import Corpus
"""
AMI class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class AMI(Corpus):
    def __init__(self, ami_folder):
        # check whether the ami_folder contains a valid AMI installation
        try:
            assert os.path.exists(ami_folder)  # folder exists
            assert os.path.exists(ami_folder+"/words/ES2002a.A.words.xml")  # words files exist
            assert os.path.exists(ami_folder+"/dialogueActs/ES2002a.A.dialog-act.xml")  # DA files exist
        except AssertionError:
            print("The folder "+ami_folder+" does not contain some important files from the corpus.")
            print("You can download a complete version of the corpus at http://groups.inf.ed.ac.uk/ami/download/")
            exit(1)
        self.ami_folder = ami_folder
        self.csv_corpus = []

    def load_csv(self):
        dialogs={}  # this will store dialogs from the corpus
        dialog_names=[]  # this will store filenames from the corpus
        for dialog_name in os.listdir(self.ami_folder+"/dialogueActs/"):
            if "dialog-act" in dialog_name:  # DA file
                dialog_names.append(dialog_name.split("dialog-act")[0])
        for dialog_name in dialog_names:
            dialogs[dialog_name] = OrderedDict()
            self.load_words(dialogs, dialog_name)
            self.load_dialog_acts(dialogs, dialog_name)
            self.load_segments(dialogs, dialog_name)
        self.csv_corpus = self.create_csv(dialogs)
        return self.csv_corpus

    def load_words(self,dialogs,dialog_name):
        with open(self.ami_folder+"/words/" + dialog_name + "words.xml") as wfile:
            for line in wfile:
                if not "<w" in line:  # not a word
                    continue
                elif "punc=" in line:  # punctuation
                    continue
                word_id = line.split("id=\"")[1].split("\"")[0].split(".")[-1]
                word_value = line.split(">")[1].split("<")[0]
                dialogs[dialog_name][word_id] = []
                dialogs[dialog_name][word_id].append(word_value)

    def load_dialog_acts(self,dialogs,dialog_name):
        with open(self.ami_folder+"/dialogueActs/" + dialog_name + "dialog-act.xml") as actfile:
            dact = ""
            for line in actfile:
                if "<nite:pointer" in line:  # act definition
                    dact = line.split("href=\"da-types.xml#id(")[1].split(")")[0]
                    continue
                elif "<nite:child" in line:  # word list for this act
                    ids = line.split("#")[1]
                    # 4.1 get the start/stopIDs to be queried
                    start_id = ids.split("..")[0].split("(")[1].split(")")[0].split("words")[1]
                    # 4.2 Get the range of IDs to be queried
                    try:
                        stop_id = ids.split("..")[1].split("(")[1].split(")")[0].split("words")[1]
                    except:
                        stop_id = start_id
                    start_n = int(start_id)
                    stop_n = int(stop_id)
                    # 4. Build the query
                    set = ["words" + str(i) for i in range(start_n, stop_n + 1)]
                    for w in set:
                        try:
                            dialogs[dialog_name][w].append(dact)
                        except:
                            continue

    def load_segments(self, dialogs, dialog_name):
        with open(self.ami_folder+"/segments/" + dialog_name + "segments.xml") as segmfile:
            segment = 0
            for line in segmfile:
                if "<nite:child" in line:  # word list for this segment
                    ids = line.split("#")[1]
                    # 4.1 get the start/stopIDs to be queried
                    start_id = ids.split("..")[0].split("(")[1].split(")")[0].split("words")[1]
                    # 4.2 Get the range of IDs to be queried
                    try:
                        stop_id = ids.split("..")[1].split("(")[1].split(")")[0].split("words")[1]
                    except:
                        stop_id = start_id
                    start_n = int(start_id)
                    stop_n = int(stop_id)
                    # 4. Build the query
                    set = ["words" + str(i) for i in range(start_n, stop_n + 1)]
                    for w in set:
                        try:
                            dialogs[dialog_name][w].append(str(segment))
                        except:
                            continue
                segment += 1

    def create_csv(self, dialogs):
        '''
        output csv:
        {filename : [(DA, utt, speaker, turn number)]}
        '''
        # print('Dialogues len: ', len(dialogs))
        # print('Dialogues type: ', type(dialogs))
        # print('Dialogues keys ex: ', dialogs.keys()[0])
        # # print('Dialogues val ex: ', dialogs[dialogs.keys()[0]])
        # # print('Dialogues val ex: ', dialogs[dialogs.keys()[1]])
        # dialogs = {k:v for k, v in dialogs.iteritems() if k==dialogs.keys()[0]} # Testing

        csv_corpus = {}
        for d in dialogs:
            csv_dialogue = []
            sentence = ""
            prevDA = "Other"
            currentDA = ""
            current_segment = -1
            turn_number = 1
            speaker_A = True # FYI: only for grid without real speakers!
            speaker = 'A'

            for word in dialogs[d]:
                try:
                    word, DA, segment = dialogs[d][word]
                except:
                    continue
                # if (DA != currentDA or current_segment != segment) and sentence != "":  # new DA or segment: print sentence
                if ((DA != currentDA and current_segment != segment) or DA != currentDA) and sentence != "":  # new DA or segment: print sentence
                    # csv_corpus.append((sentence.strip().replace("&#39;", "'"), currentDA, prevDA, segment, None, None))


                    csv_dialogue.append((currentDA,
                                         sentence.strip().replace("&#39;", "'"),
                                         speaker,
                                         turn_number))

                    sentence = ""
                    prevDA = currentDA

                # Turn change
                if current_segment != segment:
                    # print('Turn number: \n', turn_number)
                    # print('Sentence: \n', sentence)
                    if csv_dialogue!=[]:
                        turn_number += 1
                    speaker = 'A' if speaker_A == True else 'B'
                    # Change speaker
                    if speaker_A is True:
                        speaker_A = False
                    else:
                        speaker_A = True
                current_segment = segment
                currentDA = DA
                sentence = sentence + " " + (word.strip())

            csv_corpus[d] = csv_dialogue

        # print('Dialogue: ', csv_dialogue)
        return csv_corpus

    @staticmethod
    def da_to_dimension(corpus_tuple):
        da = corpus_tuple[1]
        if da in ["ami_da_4", "ami_da_5", "ami_da_6", "ami_da_7", "ami_da_8"]:
            return "Task"
        elif da in ["ami_da_1", "ami_da_9"]:
            return "Feedback"
        elif da in ["ami_da_14", "ami_da_15"]:
            return "SocialObligationManagement"
        else:
            return None

    @staticmethod
    def da_to_cf(corpus_tuple):
        da=corpus_tuple[1]
        if da == "ami_da_4":
            return "Statement"
        elif da == "ami_da_7":
            return "Commissive"
        elif da in ["ami_da_6", "ami_da_8"]:
            return "Directive"
        elif da in ["ami_da_1", "ami_da_9"]:
            return "Feedback"
        else:
            return None