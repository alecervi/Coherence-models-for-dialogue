from __future__ import print_function
from future.utils import iteritems
from builtins import dict
from corpus.Switchboard.Switchboard import Switchboard
from corpus.Oasis.Oasis import Oasis
from corpus.Maptask.Maptask import Maptask
from corpus.AMI.AMI import AMI
from collections import OrderedDict
from itertools import groupby
from operator import itemgetter
from random import shuffle
import argparse
import numpy as np
import sys
import inspect
import tqdm
import copy
import timeit
import re
import logging
import itertools
import spacy
import csv
import os

'''
GridGenerator takes as input a dialogue with DAs and converts it into a Grid,
the Grid object is then saved in a text file.

Input structure:
One DA per line

Grid structure:
     N  1  2
DA1  G  -  -
DA2  -  Q  -

Differences compared to Elsner and Charniak: they use gold parses for Switchboard,
while we give preference to gold DAs (only 53% utts of SWDA have gold parses)

- Lapata and Barzilay (2005):
Mention detection: entities are NPs, each noun in the NP given same role as head
e.g. Former Chilean dictator Augusto Pinochet mapped to 3 ents (dictator, Augusto, Pinochet)

- Barzilay and Lapata(2008):
Mention detection:
- COREF: nouns are considered coreferent if identical, NEs/multiword nouns divided and considered independent
e.g. Microsoft Corp. mapped to 2 ents (Microsoft, Corp.), role of each element in NP=head's role
+ COREF: coreferent NPs mapped to the head

- Elsner & Charniak (2011), Extending the Entity grid:
Mention detection: all NPs (not only heads), non-head nouns are given the role X

Files generated with spacy 1

 '''

all_datasets_path = '../../Datasets/'
corpora_paths = {'Switchboard': all_datasets_path + 'Switchboard/data/switchboard1-release2/',
                 'Oasis': all_datasets_path + 'Oasis',
                 'Maptask': all_datasets_path + 'maptaskv2-1',
                 'AMI': all_datasets_path + 'ami_public_manual_1.6.2'}

def get_corpus(corpus_name):
    corpus_path = corpora_paths.get(corpus_name)
    if corpus_name=='Switchboard':
        corpus_loader = Switchboard(corpus_path)
    elif corpus_name=='Oasis':
        corpus_loader = Oasis(corpus_path)
    elif corpus_name=='Maptask':
        corpus_loader = Maptask(corpus_path)
    elif corpus_name=='AMI':
        corpus_loader = AMI(corpus_path)
    corpus_dct = corpus_loader.load_csv()
    return corpus_dct, corpus_loader

class GridGenerator(object):

    def __init__(self, nlp=None, coref=None):
        if nlp is None:
            print("Loading spacy model")
            try:
                spacy.info('en_core_web_sm')
                model = 'en_core_web_sm'
            except IOError:
                print("No spacy 2 model detected, using spacy1 'en' model")
                model = 'en'
            self.nlp = spacy.load(model)

        if not coref:
            try:
                from neuralcoref_lib.neuralcoref import Coref
                self.coref = Coref(nlp=self.nlp)
            except:
                self.coref = coref
                logging.info('Coreference not available')

        else:
            self.coref = coref

        self.spacy_tags = {'pos': {'noun': ['NOUN', 'PROPN'],
                                   'pronoun': ['PRON']},
                           'dep': {'subject': ['csubj', 'csubjpass', 'nsubj', 'nsubjpass'], #'agent','expl',
                                   'object': ['dobj', 'iobj', 'oprd']}} # ADD 'attr' for spacy 2
        self.conversational_pronouns = ["i", "you"]
        self.grammatical_ranking = ['S', 'O', 'X']

    def extract_coref_from_one_shot(self, utts, speakers):
        clusters = self.coref.one_shot_coref(utterances=utts, utterances_speakers_id=speakers)
        # clusters = self.coref.get_clusters(use_no_coref_list=False) # If you want to keep "I","you"
        # print("Clusters: ", clusters)

        parsed_utts = self.coref.get_utterances(last_utterances_added=False)  # Get all spacy utts in dialogue
        entities = [[(ent.text, ent.label_) for ent in parsed_utt.ents] for parsed_utt in parsed_utts]
        # Retrieve also syntactic role

        mentions = self.coref.get_mentions()
        # print("Entities: ", entities)
        # print("Mentions: ", mentions)

        most_representative = self.coref.get_most_representative()  # With coref
        # print("Most representative: ", most_representative)
        return clusters, parsed_utts, entities, mentions

    def group_turns(self, dialogue):

        grouped = [list(g) for k, g in groupby(dialogue, itemgetter(3))]
        regrouped_dialogue = [(g[-1][0], u' '.join(sent[1] for sent in g), g[-1][2], g[-1][3])
                              for g in grouped]

        return regrouped_dialogue

    def corpus_stats(self, corpus_dct):
        corpus_dct = {dialogue_id: self.group_turns(dialogue) for dialogue_id, dialogue in iteritems(corpus_dct)}
        tokens_per_turn = [[len(turn[1].split()) for turn in dialogue] for dialogue in corpus_dct.values()]
        mean_tokens_per_turn = [sum(dialogue)/float(len(dialogue)) for dialogue in tokens_per_turn]
        mean_tokens_per_turn_all = sum(mean_tokens_per_turn) / float(len(mean_tokens_per_turn))
        turns_per_dialogue = [len(dialogue) for dialogue in corpus_dct.values()]
        mean_turns_per_dialogue = sum(turns_per_dialogue) / float(len(turns_per_dialogue))
        print("Average tokens per turn: ", mean_tokens_per_turn_all)
        print("Average dialogue turns: ", mean_turns_per_dialogue)
        return mean_tokens_per_turn_all, mean_turns_per_dialogue


    def check_named_entities(self, parsed_utt):
        NE_tokens = [[parsed_utt[i] for i in range(span.start, span.end)]
                            for span in parsed_utt.ents]
        # Strip non-nouns from NEs
        NE_tokens = self.filter_non_nouns(NE_tokens)
        return NE_tokens

    def filter_non_nouns(self, token_span):
        # Keep only nouns and pronouns (strip adjectives etc. from NPs)
        token_span = [[token for token in np if token.pos_ in self.spacy_tags['pos']['noun']+
                                                            self.spacy_tags['pos']['pronoun']]
                     for np in token_span]
        return token_span

    def extract_nps(self, parsed_utt, NEs=None,
                    include_prons=False, exclude_conversation_prons=True):

        if not NEs: NEs = []

        NPs_tokens = [[parsed_utt[i] for i in range(span.start, span.end)] for span in parsed_utt.noun_chunks]
        #  print('All NPs: ', NPs_tokens)

        # Remove all pronouns
        if include_prons is False:
            NPs_tokens = [[token for token in span]
                              for span in NPs_tokens
                              if not any(token.pos_ in self.spacy_tags['pos']['pronoun'] for token in span)]
            # print('All prons removed: ', NPs_tokens)

        # Remove only conversational pronouns
        elif exclude_conversation_prons is True:
            NPs_tokens = [[token for token in span]
                              for span in NPs_tokens
                              if not any(token.pos_ in self.spacy_tags['pos']['pronoun']
                                         and token.text.lower() in self.conversational_pronouns
                                         for token in span)]
            # print('Personal prons removed: ', NPs_tokens)

        # Strip non-nouns from NPs
        NPs_tokens = self.filter_non_nouns(NPs_tokens)
        # print("Simple NPs only nouns: ", NPs_tokens)

        # Check whether a "compound" is part of a NE, if not consider it an independent entity from head NP
        NPs_tokens = [np for np in NPs_tokens if np not in NEs]
        # print("Simple NPs no ent: ", NPs_tokens)

        return NPs_tokens

    def extract_head_nps(self, NPs_tokens=None, NEs=None):
        # Lapata(2008)
        # headNPs: (only head of Noun Phrase), partial coref (only if identical)
        if not NPs_tokens: NPs_tokens = []
        if not NEs: NEs = []

        # Keep only head from each NP (delete other tokens)
        NPs_tokens = [[token for token in span if token.head.i not in [token.i for token in span]] for span in NPs_tokens]

        # Join resulting NPs with NEs
        NPs_tokens = NPs_tokens + NEs

        return NPs_tokens


    def extract_all_nps(self, NPs_tokens=None, NEs=None):
        # Elser & Charniak (2008)
        # allNPs: (non-head mentions given role X), partial coref (only if identical)
        if not NPs_tokens: NPs_tokens = []
        if not NEs: NEs = []

        # Divide final NPs (to be considered independently)
        NPs_tokens = [[token] for np in NPs_tokens for token in np]

        # Join resulting NPs with NEs
        NPs_tokens = NPs_tokens + NEs

        return NPs_tokens


    def extract_entities_from_utt(self, utt, entities_type="headNPs", use_coref=False,
                                  include_prons= False, exclude_conversation_prons= True):

        if use_coref:
            resolved_utts = self.coref.get_resolved_utterances(use_no_coref_list=exclude_conversation_prons)
            parsed_utts = [self.nlp(resolved_utt) for resolved_utt in resolved_utts]
        else:
            parsed_utts = [self.nlp(utt)]

        # print('Parsed utt:', [u.text for u in parsed_utts])
        entities = []

        for parsed_utt in parsed_utts:
            # Extract NPs and Named Entities
            NEs = self.check_named_entities(parsed_utt)
            NPs_tokens = self.extract_nps(parsed_utt,
                                          NEs=NEs,
                                          include_prons=include_prons,
                                          exclude_conversation_prons=exclude_conversation_prons)
            # Lapata(2008)
            if entities_type=="headNPs":
                entities.append(self.extract_head_nps(NPs_tokens=NPs_tokens, NEs=NEs))

            # Elser & Charniak (2008)
            elif entities_type=="allNPs":
                entities.append(self.extract_all_nps(NPs_tokens=NPs_tokens, NEs=NEs))

        # Join Spacy utts
        entities = [ent for utt in entities for ent in utt]

        return entities


    def assign_tag_to_entities(self, entities, tag, tag_type="synrole_head"):
        # tag_type = synrole - syntactic role, if more than one in current text follow synrole ranking,
        #            da - dialogue act


        if tag_type=='da':
            entity_tags = [(entity, tag) for entity in entities]
        elif tag_type=='synrole_head':
            entity_tags = [(entity_span, "S") if any(entity.dep_ in self.spacy_tags['dep']['subject'] if entity.dep_!='compound'
                                                     else entity.head.dep_ in self.spacy_tags['dep']['subject']
                                                     for entity in entity_span)
                    else (entity_span, "O") if any(entity.dep_ in self.spacy_tags['dep']['object'] if entity.dep_!='compound'
                                                   else entity.head.dep_ in self.spacy_tags['dep']['object']
                                                   for entity in entity_span)
                    else (entity_span, "X") for entity_span in entities]
        elif tag_type=='synrole_X':
            entity_tags = [(entity_span, "S") if any(entity.dep_ in self.spacy_tags['dep']['subject'] for entity in entity_span)
                    else (entity_span, "O") if any(entity.dep_ in self.spacy_tags['dep']['object'] for entity in entity_span)
                    else (entity_span, "X") for entity_span in entities]
        elif tag_type=='is_present':
            entity_tags = [(entity_span, "X") for entity_span in entities]
        else:
            raise TypeError("Not implemented tag type")

        # print("Tagged entities: ", entity_tags)
        return entity_tags

    def group_same_utt_entities(self, current_entities, tag_type):
        grouped_entities = [list(g) for k, g in
                            groupby(sorted(current_entities), itemgetter(0))]  # Group entities by text

        grouped_entities = [[e_group[0]] if all(e[1] == e_group[0][1] for e in e_group) else e_group
                            for e_group in
                            grouped_entities]  # Reduce groups of entities with same text and same tag to one

        if all(len(e_group) == 1 for e_group in grouped_entities):
            current_entities = [e_group[0] for e_group in grouped_entities]

        # Only possible if the chosen tag_type = syntactic role
        else:
            if tag_type not in ['synrole_head','synrole_X']:
                raise TypeError('Different DA categories found in the same DA span!')

            # Get role according to rank
            get_role = lambda y: min(y, key=lambda x: self.grammatical_ranking.index(x))
            current_entities = [(e_group[0][0], get_role([x[1] for x in e_group])) if len(e_group)>1
                                else e_group[0] for e_group in grouped_entities]

        return current_entities

    def remove_disfluencies(self, current_entities):

        # Remove empty entities
        current_entities = [(entity_span, tag) for entity_span, tag in current_entities if entity_span]

        # Reduce consecutive repetitions of the same entity to one entity
        current_entities = [([entity_span[0]], tag) if all(entity.text==entity_span[0].text for entity in entity_span)
                            else (entity_span, tag) for entity_span, tag in current_entities]

        return current_entities

    def transform_tokens_groups_in_text(self, current_entities):
        # Entities are lowercased and only the text form is kept
        return [(u' '.join(ent.lower_ for ent in entity_span), tag) for entity_span, tag in current_entities]

    def complicated_coref(self, tagged_entities, previous_mentions, exclude_conversation_prons):
        all_mentions = self.coref.get_mentions()  # Tokens spans
        new_mentions = []
        # If any new mentions were found
        if (len(all_mentions) - previous_mentions) > 0:
            new_mentions = all_mentions[-(len(all_mentions) - previous_mentions):]

        print("- New Mentions: ", new_mentions)
        print("- New Mentions: ", [[(token.text, token.i) for token in span] for span in new_mentions])
        print("- All mentions: ", all_mentions)

        resolved_utt = self.coref.get_resolved_utterances()
        repr = self.coref.get_most_representative()
        clusters = self.coref.get_clusters(use_no_coref_list=exclude_conversation_prons)

        cluster_words = {all_mentions[k]: [all_mentions[v] for v in vals] for k, vals in
                         iteritems(self.coref.get_clusters())}

        print("- Clusters: ", self.coref.get_clusters())
        print("- Clusters: ", {all_mentions[k]: [all_mentions[v] for v in vals] for k, vals in
                               iteritems(clusters)})
        print("- Clusters: ", cluster_words)
        print("- RESOLVED UTT: ", resolved_utt)
        print("- Repr: ", repr)
        print("- Toks: ",
              [(coref_original[0].text, coref_original[0].pos_, coref_original[0].i) for coref_original, coref_replace
               in repr.items()])
        # print("- Toks: ", [(type(coref_original[0]), type(coref_replace[0])) for
        #                    coref_original, coref_replace in repr.items()])

        # Find mapping between mentions and current NP entities
        for entity_token_span, tag in tagged_entities:
            for mention_span in new_mentions:
                if any(entity_token in mention_span for entity_token in entity_token_span):
                    print('Found matching coreference: Entity_token_span: ', entity_token_span)
                    print('Found matching coreference: Mention_span: ', mention_span)
                    print('Found matching coreference: Mention_span index: ', all_mentions.index(mention_span))

                    # Find antecedent if there is one

                    mention_cluster = [(k, vals) for k, vals in iteritems(clusters) if
                                       all_mentions.index(mention_span) in vals]
                    print('Cluster: ', mention_cluster)
                    # print('Pairs scores: ', self.coref.get_scores()['pair_scores'])
        previous_mentions = len(all_mentions)



    def process_corpus(self, corpus, entities_type="allNPs",
                       include_prons=False, exclude_conversation_prons=True,
                       tag_type="synrole", group_by="DAspan", use_coref = False,
                       end_of_turn_tag=False, no_entity_column=False):

        '''
        :param corpus: {filename : [(DA, utt, speaker, turn number)]}
        :param entities_type = "headNPs",
                                "allNPs"
        :param group_by = "DAspan",
                        "turns"
        :param tag_type = "synrole",
                            "DAtag"

        :return: grid: {filename: [DAs[entities]]}
        # each dialogue is represented as a grid: matrix DAs x Entities (including NO ent)
        '''


        grids = {}

        # corpus = {k: v for k, v in corpus.iteritems() if k in ['sw_0657_2900.utt','sw_0915_3624.utt']} # Testing

        # For each dialogue
        for dialogue_id, dialogue in tqdm.tqdm(iteritems(corpus), total=len(corpus)):

            # Test utt
            # print("Dialogue id: ", dialogue_id)
            # print("Dialogue len: ", len(dialogue))
            # dialogue = dialogue[:8]
            # test_utt = u"San Francisco is a great town. She loves it. It is great, do you agree? " \
            #            u"The world's largest oil company is located there. Drugs are great! The chilean leader Barack Obama was happy."
            # test_utt = u"My mom is great. I love her! I love drugs!"
            # test_utt2 = u"I also like them. The world's largest oil company is here. My mom is called Julia."
            # # test_utt = u'Hello!'
            # dialogue[0] = (u'test', test_utt, u'A', -2)
            # dialogue[1] = (u'test', test_utt2, u'B', -1)

            if self.coref is not None:
                self.coref.clean_history()

            # previous_mentions = 0
            dialogue_entities = {} # List of turns (entity: list of turns)
            if no_entity_column is True:
                dialogue_entities['no_entity'] = []

            # Select text span
            if group_by=="turns":
                dialogue = self.group_turns(dialogue)

            # Minimum 5 dialogue turns
            if len(self.group_turns(dialogue)) < 5:
                continue

            # For each text span (utterance) extract list of entities
            for tag, utt, speaker, turn_id in dialogue:

                previous_turns_len = len(list(dialogue_entities.values())[0]) if dialogue_entities.keys() else 0

                # Preprocess utt removing double spaces
                utt = re.sub(' +', ' ', utt)

                # start = timer()
                if use_coref:
                    self.coref.continuous_coref(utterances=utt, utterances_speakers_id=speaker)

                # t_continuous_coref = timer()
                # print('Time only continuous_coref: ', t_continuous_coref - start)

                # Extract entities
                current_entities = self.extract_entities_from_utt(utt,
                                                                  entities_type=entities_type,
                                                                  include_prons=include_prons,
                                                                  use_coref=use_coref,
                                                                  exclude_conversation_prons=exclude_conversation_prons)

                # Assign tag to the entities: [(token list, tag)]
                tagged_entities = self.assign_tag_to_entities(current_entities, tag, tag_type=tag_type)

                # Remove disfluencies
                tagged_entities = self.remove_disfluencies(tagged_entities)

                # Transform spacy tokens into text
                tagged_entities = self.transform_tokens_groups_in_text(tagged_entities)

                # Map repetitions of the same entity in current utt
                tagged_entities = self.group_same_utt_entities(tagged_entities, tag_type)

                # Check previous entity dict if there is already an entity, else add new entity column
                for entity_key, entity_tag in tagged_entities:

                    if entity_key in dialogue_entities:
                        dialogue_entities[entity_key].append(entity_tag) # Update entity with new tag
                    else:
                        dialogue_entities[entity_key] = ['_']*previous_turns_len+[entity_tag]


                # Update no entity column if there are no entities
                if not tagged_entities and no_entity_column and tag_type=='da':
                    dialogue_entities['no_entity'].append(tag)  # Update entity with new tag
                #  Initialize dialogue_entities if there are no tagged entities creating tmp entity later to be dropped
                elif not tagged_entities and not dialogue_entities:
                    dialogue_entities['<tmp_none>']=['_']


                # Update all entities not in this turn
                dialogue_entities = {ent: (tags+['_'] if len(tags)<(previous_turns_len+1) else tags)
                                     for ent, tags in iteritems(dialogue_entities)}

                if end_of_turn_tag is True:
                    dialogue_entities = {ent: (tags + ['<eot>'])
                                         for ent, tags in iteritems(dialogue_entities)}
                # t_end = timer()
                # print('Time from continuous_coref to end of dialogue: ', t_end - t_continuous_coref)
                # print('Grids lengths: ', set([len(en) for en in dialogue_entities.values()]))

                # print('--Grid -Whole grid: ', dialogue_entities)

            # Remove tmp initializing entity
            dialogue_entities.pop('<tmp_none>', None)

            grids[dialogue_id] = dialogue_entities

            # break

        logging.info('All grids parsed')


        return grids


    #
    # def get_intra_shuffle(self, dialogue, shuffles_number=5):
    #     index_shuff = range(len(dialogue))
    #     shuffled_orders = [shuffle(index_shuff) for shuff_i in range(shuffles_number)]
    #     return shuffled_orders





    def sort_grid_entity_appearance(self, grids_dct):
        return OrderedDict(sorted(grids_dct.items(), key=lambda x: min(i for i, v in enumerate(x[1]) if v != "-")))

    def turn_grids_into_to_write(self, grids_dct):
        # print('Formatted grids values 0: ', grids_dct.values()[0])
        formatted = [[entity[i] for entity in grids_dct.values()] for i in range(len(list(grids_dct.values())[0]))]
        formatted.insert(0, grids_dct.keys())
        return formatted


    def create_csv_folder(self, grids_dct, options, folder_name, folder_path, corpus_name, min_len_dial = 1):
        full_path = folder_path + corpus_name +'/'+folder_name+'/'
        logging.info('Creating output directory: %s', full_path)

        if not os.path.exists(full_path):
            os.makedirs(full_path)
        empty_dials = []
        dump_csv(full_path+'Params', options.items()) # Params file
        logging.info('Params file created')
        for dialogue_id, dialogue in iteritems(grids_dct):
            if len(dialogue) >= min_len_dial:

                formatted_grid = self.sort_grid_entity_appearance(grids_dct[dialogue_id])
                formatted_grid = self.turn_grids_into_to_write(formatted_grid)
                dump_csv(full_path+dialogue_id, formatted_grid)
            else:
                empty_dials.append(dialogue_id)
        print('Empty dialogue id: ', empty_dials)
        print('Len Empty dialogue id: ', len(empty_dials))
        return


def dump_csv(out_file, to_write):
    # print(to_write)
    with open(out_file + '.csv', 'w') as out:
        csv_out = csv.writer(out)
        for row in to_write:
            csv_out.writerow(row)

def main(args):
    print(''.join(y for y in["-"]*180))
    logging.basicConfig(
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format='%(levelname)s %(message)s')

    if not args.outputname:
        raise TypeError('Missing output directory name')
    # if os.path.exists(args.outputpath+args.outputname):
    #     raise Warning('Folder %s already exists', args.outputpath+args.outputname)
    #     overwrite_folder = raw_input("Enter your name: ")


    # swda = Switchboard(args.input)
    # corpus_dct = swda.load_csv()

    corpus_dct, corpus_loader = get_corpus(args.input)

    logging.info('Files number: %d', len(corpus_dct))
    # tags = corpus_loader.get_tags()
    # logging.info('DAs number: %d', len(tags))
    logging.info('Corpus loaded')
    logging.info('Corpus dimension: %d', len(corpus_dct))

    # Debug
    print('Ex. File names in dct: ', list(corpus_dct.keys())[0])
    print('Ex. Dialogue type: ', type(corpus_dct[list(corpus_dct.keys())[0]]))
    print('Ex. Len of dialogue: ', len(corpus_dct[list(corpus_dct.keys())[0]]))
    print('Ex. Turn type: ', type(corpus_dct[list(corpus_dct.keys())[0]][0]))
    print('Ex. Turn 0: ', corpus_dct[list(corpus_dct.keys())[0]][0])
    print('Ex. Turn 0-4: ')
    for y in corpus_dct[list(corpus_dct.keys())[0]]:
        print(y)

    grid_generator = GridGenerator()

    options = {
        'entities_type' : 'headNPs',
        'include_prons' : False,
        'exclude_conversation_prons' : True,
        'group_by' : 'DAspan', # DAspan, turns (Elsner & Charniak, 2011, "Disentangling Chat")
        'tag_type' : 'da', # da, synrole_head (Lapata & Barzilay 2005/2008), synrole_X (Elsner & Charniak 2011)
        'use_coref' : False,
        'no_entity_column' : True,
        'end_of_turn_tag' : False}

    default_confs = ['egrid_-coref', 'egrid_+coref', 'extgrid_-coref', 'simple_egrid_-coref',
                     'egrid_-coref_DAspan', 'egrid_-coref_DAspan_da', 'egrid_-coref_DAspan_da_noentcol']

    if args.default not in default_confs+['']: raise TypeError('Default configuration inserted is not allowed')

    if args.default in default_confs:
        logging.info('Default configuration requested: %s', args.default)
        options['group_by'] = 'turns'
        options['no_entity_column'] = False
        options['end_of_turn_tag'] = False

        if args.default in ['egrid_-coref', 'egrid_-coref_DAspan', 'egrid_-coref_DAspan_da',
                            'egrid_-coref_DAspan_da_noentcol', 'simple_egrid_-coref']:
            # Head noun divided into, each noun in NP same grammatical role
            options['entities_type'] = 'allNPs'
            options['tag_type'] = 'synrole_head'
            options['use_coref'] = False
            if args.default in ['simple_egrid_-coref']:
                options['tag_type'] = 'is_present'
            if args.default in ['egrid_-coref_DAspan', 'egrid_-coref_DAspan_da','egrid_-coref_DAspan_da_noentcol']:
                options['group_by'] = 'DAspan'
            if args.default in ['egrid_-coref_DAspan_da','egrid_-coref_DAspan_da_noentcol']:
                options['tag_type'] = 'da'
            if args.default in ['egrid_-coref_DAspan_da_noentcol']:
                options['no_entity_column'] = True

        elif args.default=='egrid_+coref':
            # Keep only head nouns, perform coreference on it
            options['entities_type'] = 'headNPs'
            options['tag_type'] = 'synrole_head'
            options['use_coref'] = True
            options['include_prons'] = True

        # Extended grid default: what's the difference with egrid_-coref? Supposedly "Bush spokeman", where Bush=X
        elif args.default=='extgrid_-coref':
            # Add non-head nouns
            options['entities_type'] = 'allNPs'
            options['tag_type'] = 'synrole_X'
            options['use_coref'] = False

    logging.info('Set up')
    for k, v in iteritems(options):
        logging.info('%s : %s', k, v)

    # Process corpus
    grids = grid_generator.process_corpus(corpus_dct,
                                          entities_type=options['entities_type'],
                                          include_prons=options['include_prons'],
                                          exclude_conversation_prons= options['exclude_conversation_prons'],
                                          group_by=options['group_by'],
                                          tag_type=options['tag_type'],
                                          use_coref=options['use_coref'],
                                          no_entity_column=options['no_entity_column'],
                                          end_of_turn_tag=options['end_of_turn_tag'])

    print('Len grids: ', len(grids))

    # Write out
    grid_generator.create_csv_folder(grids, options, args.outputname, args.outputpath, args.input)





def argparser(parser=None, func=main):
    """parse command line arguments"""

    if parser is None:
        parser = argparse.ArgumentParser(prog='grid')

    parser.description = 'Generative implementation of Entity grid'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('input', nargs='?',
                        # type=argparse.FileType('r'),
                        # default=sys.stdin,
                        type=str,
                        help='input corpus in doctext format')

    parser.add_argument('default', nargs='?',
                        type=str,
                        default='',
                        help="default settings of all params")

    parser.add_argument('outputname', nargs='?',
                        type=str,
                        help="output_name")

    parser.add_argument('outputpath', nargs='?',
                        type=str,
                        default='data/',
                        help="output folder path")

    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help='increase the verbosity level')

    if func is not None:
        parser.set_defaults(func=func)

    return parser

if __name__ == '__main__':
    main(argparser().parse_args())

# Old
# python generate_grid.py ../../Datasets/Switchboard/data/switchboard1-release2/ outout
# python generate_grid.py ../../Datasets/Switchboard/data/switchboard1-release2/ egrid_-coref_DAspan_da egrid_-coref_DAspan_da data/ -v
# python generate_grid.py ../../Datasets/Switchboard/data/switchboard1-release2/ egrid_-coref_DAspan_da_noentcol egrid_-coref_DAspan_da_noentcol data/ -v

# New
# python generate_grid.py Switchboard egrid_-coref_DAspan_da egrid_-coref_DAspan_da data/ -v
# python generate_grid.py Oasis egrid_-coref_DAspan_da egrid_-coref_DAspan_da data/ -v
# python generate_grid.py AMI egrid_-coref egrid_-coref data/ -v
# python generate_grid.py Switchboard simple_egrid_-coref simple_egrid_-coref data/ -v
# simple_egrid_-coref