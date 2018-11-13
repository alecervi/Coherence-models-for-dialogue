from __future__ import print_function
from __future__ import unicode_literals
from builtins import dict
import os
import csv
import re
from .DAMSL import DAMSL
from corpus.Corpus import Corpus

"""
Switchboard class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class Switchboard(Corpus):
    def __init__(self, corpus_folder):
        try:
            assert os.path.exists(corpus_folder)  # folder exists
            assert os.path.exists(corpus_folder + "/sw00utt")  # dialogs folders exist
            assert os.path.exists(corpus_folder + "/sw00utt/sw_0001_4325.utt")  # DA files exist
        except AssertionError:
            print("The folder " + corpus_folder + " does not contain some important files from the corpus.")
            print("Check https://catalog.ldc.upenn.edu/ldc97s62 for info on how to obtain the complete SWDA corpus.")
            exit(1)
        self.corpus_folder = corpus_folder
        self.csv_corpus = []
        self.tags_list = []

    def load_csv(self):
        # Read dialogue files from Switchboard
        filelist=self.create_filelist()
        self.csv_corpus=self.create_dialogue_csv(filelist)
        self.update_tags()
        return self.csv_corpus


    @staticmethod
    def get_regex(all=False, keys=["default"]):
        regexes = {"default":r"([+/\}\[\]]|\{\w)", # this REGEX removes prosodic information and disfluencies
                   "double_round": r"\({2}|\){2}",
                   "square_comments": r"\*\[{2}\w+(\s\w+)*\]{2}",
                   "angular_comments": r"\<\w+(\s\w+)*\>",
                   "slashes": r"\-{2}|[^a-z]\-{1}[^a-z]|\w+\-{1}[^a-z]",
                   "hash": r"#",
                   "typo": r"\*typo"
                   }
        result_regex = ""
        if all is True:
            result_regex = "|".join(regex for regex in regexes.values())
        else:
            if any(regex_type not in regexes.keys() for regex_type in keys):
                raise TypeError("Invalid regex type requested")
            result_regex = "|".join(regexes.get(regex_type) for regex_type in keys)
        return result_regex

    @staticmethod
    def write_csv(to_write, headers, filename="generated_dataset", outpath=""):
        with open(outpath+filename+'.csv','wb') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerow(headers)
            for line in to_write:
                writer.writerow(line)
        print("Written output csv file: ", filename)

    def create_filelist(self):
        filelist=[]
        for folder in os.listdir(self.corpus_folder):
            if folder.startswith("sw"):  # dialog folder
                for filename in os.listdir(self.corpus_folder+"/"+folder):
                    if filename.startswith("sw"):  # dialog file
                        filelist.append(self.corpus_folder+"/"+folder+"/"+filename)
        return filelist

    def create_csv(self,filelist):
        csv_corpus = []
        for filename in filelist:
            prev_speaker = None
            segment = 0
            prev_DAs = {"A":"%", "B":"%"}
            with open(filename) as f:
                utterances = f.readlines()
            for line in utterances:
                line = line.strip()
                try:
                    sentence = line.split("utt")[1].split(":")[1]
                    sw_tag = line.split("utt")[0].split()[0]
                    if "A" in line.split("utt")[0]:  # A speaking
                        speaker = "A"
                    else:
                        speaker = "B"
                except:  # not an SWDA utterance format: probably a header line
                    continue
                if speaker != prev_speaker:
                    prev_speaker = speaker
                    segment += 1
                sentence = re.sub(r"([+/\}\[\]]|\{\w)", "",
                                sentence)  # this REGEX removes prosodic information and disfluencies
                DA_tag = DAMSL.sw_to_damsl(sw_tag, prev_DAs[speaker])
                csv_corpus.append((sentence, DA_tag, prev_DAs[speaker], segment, None, None))
                prev_DAs[speaker] = DA_tag
        return csv_corpus

    def create_dialogue_csv(self, filelist):
        '''
        output csv:
        {filename : [(DA, utt, speaker, turn number)]}
        '''
        csv_corpus = {}
        # filelist = filelist[:5]
        # filelist = ['../../Datasets/Switchboard/data/switchboard1-release2//sw00utt/sw_0004_4327.utt']

        for filename in filelist:
            csv_dialogue = []
            prev_speaker=None
            segment=0
            prev_DAs={"A":"%","B":"%"}
            with open(filename) as f:
                utterances = f.readlines()
            for line in utterances:
                line = line.strip()
                try:
                    sentence = line.split("utt")[1].split(":")[1]
                    sw_tag = line.split("utt")[0].split()[0]
                    if "A" in line.split("utt")[0]:  # A speaking
                        speaker="A"
                    else:
                        speaker="B"
                except:  # not an SWDA utterance format: probably a header line
                    continue
                if speaker != prev_speaker:
                    prev_speaker = speaker
                    segment += 1
                sentence = re.sub(self.get_regex(all=True), "", sentence)
                sentence = re.sub(r"\'re[^\w]", " are", sentence) # preprocess "lawyers're" into "lawyers are"
                DA_tag = DAMSL.sw_to_damsl(sw_tag,prev_DAs[speaker])
                csv_dialogue.append((DA_tag, sentence, speaker, segment))
            csv_corpus[filename.split("/")[-1]] = csv_dialogue

        return csv_corpus




    @staticmethod
    def da_to_dimension(corpus_tuple):
        da=corpus_tuple[1]
        if da in ["statement-non-opinion","statement-opinion","rhetorical-questions","hedge","or-clause",
                  "wh-question", "declarative-wh-question","backchannel-in-question-form","yes-no-question",
                  "declarative-yn-question","tag-question","offers-options-commits","action-directive"]:
            return "Task"
        elif da in ["thanking","apology","downplayer","conventional-closing"]:
            return "SocialObligationManagement"
        elif da in ["signal-non-understanding", "acknowledge", "appreciation"]:
            return "Feedback"
        else:
            return None

    @staticmethod
    def da_to_cf(corpus_tuple):
        da=corpus_tuple[1]
        if da in ["statement-non-opinion","statement-opinion","rhetorical-questions","hedge"]:
            return "Statement"
        elif da == "or-clause":
            return "ChoiceQ"
        elif da in ["wh-question","declarative-wh-question"]:
            return "SetQ"
        elif da in ["backchannel-in-question-form","yes-no-question","declarative-yn-question","tag-question"]:
            return "PropQ"
        elif da == "offers-options-commits":
            return "Commissive"
        elif da == "action-directive":
            return "Directive"
        elif da in ["thanking"]:
            return "Thanking"
        elif da in ["apology","downplayer"]:
            return "Apology"
        elif da in "conventional-closing":
            return "Salutation"
        elif da in ["signal-non-understanding","acknowledge","appreciation"]:
            return "Feedback"
        else:
            return None