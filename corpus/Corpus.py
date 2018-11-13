import csv
from builtins import dict


class Corpus:
    def __init__(self, corpus_folder):
        raise NotImplementedError()

    def load_csv(self):
        raise NotImplementedError()

    def create_csv(self, dialogs):
        raise NotImplementedError()


    def update_tags(self):
        self.tags_list = list(set([turn[0] for conversation in self.csv_corpus.values() for turn in conversation]))

    def get_tags(self):
        # print self.tags_list
        return self.tags_list

    def dump_csv(self,out_file):
        with open(out_file, 'wb') as out:
            csv_out = csv.writer(out)
            for row in self.csv_corpus:
                csv_out.writerow(row)

    def dump_iso_dimension_csv(self, out_file):
        with open(out_file, 'wb') as out:
            csv_out = csv.writer(out)
            for row in self.csv_corpus:
                if row is not None:
                    csv_out.writerow(self.corpus_tuple_to_iso_dimension(row))

    def dump_iso_task_csv(self, out_file):
        with open(out_file, 'wb') as out:
            csv_out = csv.writer(out)
            for row in self.csv_corpus:
                if row is not None:
                    csv_out.writerow(self.corpus_tuple_to_iso_task(row))

    def dump_iso_som_csv(self, out_file):
        with open(out_file, 'wb') as out:
            csv_out = csv.writer(out)
            for row in self.csv_corpus:
                if row is not None:
                    csv_out.writerow(self.corpus_tuple_to_iso_som(row))

    def dump_iso_fb_csv(self, out_file):
        with open(out_file, 'wb') as out:
            csv_out = csv.writer(out)
            for row in self.csv_corpus:
                if row is not None:
                    csv_out.writerow(self.corpus_tuple_to_iso_fb(row))

    @staticmethod
    def da_to_dimension(corpus_tuple):
        raise NotImplementedError()

    @staticmethod
    def da_to_cf(corpus_tuple):
        raise NotImplementedError()

    def corpus_tuple_to_iso_dimension(self, corpus_tuple):
        da = self.da_to_dimension(corpus_tuple)
        prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
        if prevDA is None:
            prevDA="Other"
        if da is not None:
            return tuple([corpus_tuple[0]]+[da, prevDA] + corpus_tuple[2:])
        else:
            return None

    def corpus_tuple_to_iso_task(self, corpus_tuple):
        if self.da_to_dimension(corpus_tuple) == "Task":
            da = self.da_to_cf(corpus_tuple)
            prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
            if prevDA is None:
                prevDA = "Other"
            if da is not None:
                return tuple([corpus_tuple[0]]+[da, prevDA] + corpus_tuple[2:])
            else:
                return None

    def corpus_tuple_to_iso_som(self, corpus_tuple):
        if self.da_to_dimension(corpus_tuple) == "SocialObligationManagement":
            da = self.da_to_cf(corpus_tuple)
            prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
            if prevDA is None:
                prevDA = "Other"
            if da is not None:
                return tuple([corpus_tuple[0]]+[da, prevDA] + corpus_tuple[2:])
            else:
                return None

    def corpus_tuple_to_iso_fb(self, corpus_tuple):
        if self.da_to_dimension(corpus_tuple) == "Feedback":
            da = self.da_to_cf(corpus_tuple)
            prevDA = self.da_to_cf((None, corpus_tuple[2], None, None, corpus_tuple[5], None))
            if prevDA is None:
                prevDA = "Other"
            if da is not None:
                return tuple([corpus_tuple[0]]+[da, prevDA] + corpus_tuple[2:])
            else:
                return None