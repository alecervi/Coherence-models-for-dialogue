from __future__ import print_function
from __future__ import unicode_literals
import os
import csv
from lxml import etree
from corpus.Corpus import Corpus
"""
Oasis class: loads the corpus into tuples (sentence,DA,prevDA). Provides methods
to dump the corpus in CSV format with original annotation and with ISO annotation
"""


class Oasis(Corpus):
    def __init__(self, oasis_folder):
        # check whether the oasis_folder contains a valid Oasis installation
        try:
            assert os.path.exists(oasis_folder)  # folder exists
            assert os.path.exists(oasis_folder + "/Data/Lancs_BT150")  # dialogs folders exist
            assert os.path.exists(oasis_folder + "/Data/Lancs_BT150/075812009.a.lturn.xml")  # DA files exist
        except AssertionError:
            print("The folder " + oasis_folder + " does not contain some important files from the corpus.")
            print("Check http://groups.inf.ed.ac.uk/oasis/ for info on how to obtain the complete SWDA corpus.")
            exit(1)
        self.oasis_folder = oasis_folder
        self.csv_corpus = []

    def load_csv(self):
        # Read dialogue files from Oasis
        dialogs=self.create_dialogs()
        self.csv_corpus = self.create_csv(dialogs)
        return self.csv_corpus

    def create_dialogs(self):
        dialogs = {}
        for fname in os.listdir(self.oasis_folder+"/Data/Lancs_BT150/"):
            f = open(self.oasis_folder+"/Data/Lancs_BT150/" + fname.strip())
            t = etree.parse(f)
            turns = t.xpath("//lturn")
            for turn in turns:
                self.parse_xml_turn(dialogs, turn)
        return dialogs

    def parse_xml_turn(self, dialogs, turn):
        dialog_id = turn.attrib["id"].split(".")[0]
        try:  # subturn
            turn_id = int(turn.attrib["id"].split(".")[-2])
        except:  # turn
            turn_id = int(turn.attrib["id"].split(".")[-1])
        if dialogs.get(dialog_id, None) is None:  # new dialog
            dialogs[dialog_id] = {}
        if dialogs[dialog_id].get(turn_id, None) is None:  # new turn
            dialogs[dialog_id][turn_id] = []
        segments = turn.xpath(".//segment")
        for segment in segments:
            self.add_segment_to_dialog(dialogs, dialog_id, turn_id, segment)

    def add_segment_to_dialog(self,dialogs, dialog_id, turn_id, segment):
        segm_type = segment.attrib["type"]
        tag = segment.attrib["sp-act"]
        try:
            wFile = segment[0].attrib["href"].split("#")[0]
        except:
            return
        ids = segment[0].attrib["href"].split("#")[1]
        start_id = ids.split("(")[1].split(")")[0]
        stop_id = ids.split("(")[-1][:-1]
        start_n = int(start_id.split(".")[3])
        text = wFile.split(".xml")[0]
        if not 'anchor' in stop_id:
            stop_n = int(stop_id.split(".")[3])
        else:
            stop_n = start_n
        id_set = ["@id = '" + text + "." + str(i) + "'" for i in range(start_n, stop_n + 1)]
        with open(self.oasis_folder+"/Data/Lancs_BT150/" + wFile) as f:
            tree = etree.parse(f)
            segment = tree.xpath('//*[' + " or ".join(id_set) + ']')
            sentence = " ".join([x.text for x in segment if
                                 x.text is not None and x.text not in ["?", ",", ".", "!", ";"]])
            if sentence != "":
                dialogs[dialog_id][turn_id].append((sentence, tag, segm_type))

    def create_csv(self, dialogs):
        '''
        output csv:
        {filename : [(DA, utt, speaker, turn number)]}
        '''
        # print('Dialogues len: ', len(dialogs))
        # print('Dialogues type: ', type(dialogs))
        # print('Dialogues keys: ', dialogs.keys()[0])
        # print('Dialogues val ex: ', dialogs[dialogs.keys()[0]])
        # print('Dialogues keys: ', dialogs.keys()[1])
        # print('Dialogues val ex: ', dialogs[dialogs.keys()[1]])
        csv_corpus = {}
        for d in dialogs:
            csv_dialogue = []
            prevTag = "other"
            prevType = "other"
            speaker_A = True
            turn_number = 1

            for segm in sorted(dialogs[d].keys()):
                speaker = 'A' if speaker_A==True else 'B'

                for sentence in dialogs[d][segm]:

                    # csv_corpus.append((sentence[0], sentence[1], prevTag, segm, sentence[2], prevType))
                    # csv_dialogue.append((sentence[1], unicode(sentence[0], "utf-8"), speaker, turn_number)) # Python 2.7
                    csv_dialogue.append((sentence[1], sentence[0], speaker, turn_number))


                # Avoid indexes for empty turns
                if dialogs[d][segm]:
                    turn_number += 1

                # Change speaker
                if speaker_A is True:
                    speaker_A = False
                else:
                    speaker_A = True

                try:
                    prevTag = dialogs[d][segm][-1][1]
                    prevType = dialogs[d][segm][-1][2]
                except:  # no prev in this segment
                    pass

            csv_corpus[d] = csv_dialogue
        return csv_corpus

    @staticmethod
    def da_to_dimension(corpus_tuple):
        da=corpus_tuple[1]
        da_type=corpus_tuple[4]
        if da in ["suggest","inform","offer"] or da_type in ["q_wh","q_yn","imp"]:
            return "Task"
        elif da in ["thank","bye","greet","pardon","regret"]:
            return "SocialObligationManagement"
        elif da =="ackn" or da_type=="backchannel":
            return "Feedback"
        else:
            return None

    @staticmethod
    def da_to_cf(corpus_tuple):
        da=corpus_tuple[1]
        da_type=corpus_tuple[4]
        if da_type == "q_wh":
            return "SetQ"
        elif da_type == "q_yn":
            return "CheckQ"
        elif da_type=="imp" or da=="suggest":
            return "Directive"
        elif da=="inform":
            return "Statement"
        elif da=="offer":
            return "Commissive"
        elif da=="thank":
            return "Thanking"
        elif da in ["bye","greet"]:
            return "Salutation"
        elif da in ["pardon","regret"]:
            return "Apology"
        elif da=="ackn" or da_type=="backchannel":
            return "Feedback"
        else:
            return None