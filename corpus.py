import argparse
import configargparse
from typing import Iterable, List, MutableSet, Optional

from torch.utils.data.dataset import Dataset

from features import *
from morpheme import *
from tokenizer import *


class MorphemeCorpus(Dataset):

    def __init__(self, *,
                 alphabet: Alphabet,
                 sentences: Iterable[str],
                 tokenizer: Tokenizer,
                 blacklist_char: str,
                 start_of_morpheme: str, end_of_morpheme: str,
                 max_graphemes_per_morpheme: Optional[int]):

        morphemes: List[List[str]] = list()
        processed_morphemes: MutableSet[str] = set()

        for sentence in sentences:
            for word in tokenizer.words(sentence):
                if not word.startswith(blacklist_char):
                    for morpheme in tokenizer.morphemes(word):
                        if morpheme not in processed_morphemes:
                            processed_morphemes.add(morpheme)
                            graphemes = tokenizer.graphemes(morpheme)
                            if max_graphemes_per_morpheme is None or len(graphemes) <= max_graphemes_per_morpheme:
                                morphemes.append(graphemes)

        self.morphemes = Morphemes(alphabet=alphabet,
                                   start_of_morpheme=start_of_morpheme, end_of_morpheme=end_of_morpheme,
                                   list_of_morphemes=morphemes)

    def __len__(self) -> int:
        return len(self.morphemes)

    def __getitem__(self, index: int) -> Morpheme:
        return self.morphemes[index]

    @staticmethod
    def load(filename: str) -> 'MorphemeCorpus':
        import pickle
        with open(filename, 'rb') as pickled_file:
            return pickle.load(pickled_file)

    def dump(self, filename: str) -> None:
        import pickle
        with open(filename, 'wb') as pickled_file:
            pickle.dump(self, pickled_file)


def configure(arguments: List[str]) -> argparse.Namespace:

    p = configargparse.get_argument_parser(description="Construct corpus tokenizer")
    p.add('-c', '--config', required=False, is_config_file=True, type=str, metavar='FILENAME',
          help='configuration file')

    p.add('-a', '--alphabet', required=True, type=str, metavar='FILENAME',
          help='Pickle file containing a Alphabet object')

    p.add('--tokenizer', required=True, type=str, metavar='FILENAME',
          help='Pickle file containing a Tokenizer object')

    p.add('-b', '--blacklist_character', required=True, type=str, metavar='STRING',
          help="In the user-provided input file, words that begin with this character will be ignored." +
               "This symbol should not appear in the alphabet")

    p.add('-m', '--max_characters', required=False, type=int, metavar='N',
          help='Maximum number of characters allowed per morpheme (not including reserved start and end characters). ' +
               'If not provided, value will be inferred from longest morpheme in input.')

    p.add('--start_of_morpheme', required=True, type=str, metavar='STRING',
          help="Reserved symbol to be inserted at the start of each morpheme. This symbol must appear in the alphabet")

    p.add('--end_of_morpheme', required=True, type=str, metavar='STRING',
          help="Reserved symbol to be inserted at the end of each morpheme. This symbol must appear in the alphabet")

    p.add('-i', '--input_file', required=True, type=str, metavar="FILENAME",
          help="Input file containing corpus in plain-text format")

    p.add('-o', '--output_file', required=True, type=str, metavar="FILENAME",
          help="Output file where pickled MorphemeCorpus object will be saved")

    return p.parse_args(args=arguments)


def main(args: argparse.Namespace) -> None:

    import pickle

    with open(args.input_file, 'rt') as input_file, open(args.output_file, 'wb') as output_file:

        corpus = MorphemeCorpus(alphabet=Alphabet.load(args.alphabet),
                                tokenizer=Tokenizer.load(args.tokenizer),
                                sentences=input_file,
                                blacklist_char=args.blacklist_character,
                                max_graphemes_per_morpheme=args.max_graphemes_per_morpheme,
                                start_of_morpheme=args.start_of_morpheme,
                                end_of_morpheme=args.end_of_morpheme)

        for morpheme in corpus.morphemes:
            print(f"{str(morpheme)}\tlen={len(morpheme)}", file=sys.stderr)

        pickle.dump(corpus, output_file)


if __name__ == "__main__":

    import sys

    main(configure(arguments=sys.argv[1:]))
