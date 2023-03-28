import spacy
import re
import pickle
import numpy as np
import os

nlp = spacy.load('en_core_web_sm')

token_dictionary = {
    'gentive_token': '__genitive__',
    'comma_token': "__comma__",
    "question_token": "__question__",
    "delim_token": "__delim__",
    'sep_token': "__sep__",
    'line_token': "__line__",
    'triple_dot_token': "__triple_dot__",
    'start_token': "__start__",
    'end_token': "__end__"
}

class SignHandler:
    def __init__(self) -> None:
        pass
    def __sign(self, sequence: str) -> str:
        sequence = re.sub(r"\.", f" . ", sequence)
        sequence = re.sub(r'\?', f" ? ", sequence)
        sequence = re.sub(r"\,", f" , ", sequence)
        sequence = re.sub("\\n", fr" new_line__ ", sequence)
        sequence = re.sub(r"\(", fr" ( ", sequence)
        sequence = re.sub(r"\)", fr" ) ", sequence)

        """ arr = sequence.split(' ')
        result = []
        for item in arr:
            if item == '':
                continue
            if item[0].isupper():
                result.append("upper__")
            result.append(item)
        
        return " ".join(result).lower() """

 

        return sequence

    def sign(self, sequences: list, start_token: bool, end_token: bool) -> str:
        for index in range(len(sequences)):
            sequences[index] = self.__sign(sequence=sequences[index])
            if start_token:
                sequences[index] = f"{token_dictionary['start_token']} {sequences[index]}"
            if end_token:
                sequences[index] = f"{sequences[index]} {token_dictionary['end_token']}"
        return sequences



class Cleanner:
    def __init__(self, filters: str) -> None:
        self.filters = filters
    def __clean(self, sequence: str) -> str:
        sequence = sequence.lower()
        sequence = sequence.strip()
        sequence = re.sub(r'[.,!]', "", sequence)
        sequence = re.sub('\s\s+', ' ', sequence)
        return sequence

    def clean(self, sequences: list) -> list:
        for index in range(len(sequences)):
            sequences[index] = self.__clean(sequence=sequences[index])
        return sequences


class Remover:
    def __init__(self) -> None:
        self.stop_words = nlp.Defaults.stop_words
    def __remove(self, sequence: str) -> str:
        chars = sequence.split(' ')
        tokens = [token for token in chars if not token in self.stop_words]

        return " ".join(tokens)

    def remove(self, sequences: list) -> list:
        for index in range(len(sequences)):
            sequences[index] = self.__remove(sequence=sequences[index])

        return sequences

class Replacer:
    def __init__(self) -> None:
        self.replace_patterns = [
            (r"won\'t", 'will not'),
            (r"'t", 'cannot'),
            (r"'ll", "will"),
            (r"n\'t", "not"),
            (r"'ve", "have"),
            (r"i\'m", 'i am'),
            (r"'re", "are")
        ]

    def __replace_standard(self, sequence: str) -> str:
        for (pattern, repl) in self.replace_patterns:
            sequence = re.sub(pattern, repl, sequence)

        return sequence


    def replace(self, sequences: list) -> list:
        for index in range(len(sequences)):
            sequences[index] = self.__replace_standard(sequence=sequences[index])
        return sequences



    def __lemma(self, sequence: str) -> str:
        parts = sequence.split('delim__')
        result1 = self.anlysis(parts[0])
        if len(parts) != 1:
            result2 = self.anlysis(parts[1])
        else:
            result2 = ""
        return f"{result1} delim__ {result2}"

    def lemma(self, sequences: list) -> list:
        for index in range(len(sequences)):
            sequences[index] = self.__lemma(sequence=sequences[index])

        return sequences


class Tokenizer:
    def __init__(self, tokenizer_path: str = None) -> None:
        self.tokenizer_path = tokenizer_path
        self.token_index = dict()
        self.index_token = dict()
        self.token_counts = dict()
        self.num_tokens = 0

    def __save_tokenizer(self, tokenizer_path: str = None) -> None:
        if tokenizer_path is not None:
            self.tokenizer_path = tokenizer_path
        with open(f'{self.tokenizer_path}', 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer is saved at {self.tokenizer_path}")

    def __load_tokenizer(self) -> None:
        if self.tokenizer_path is None:
            return
        if os.path.exists(f"{self.tokenizer_path}") == False:
            return
        with open(f'{self.tokenizer_path}', 'rb') as file:
            self = pickle.load(file)

    def add_token(self, token: str) -> None:
        if token not in self.token_index:
            self.num_tokens += 1
            self.token_index[token] = self.num_tokens
            self.index_token[self.num_tokens] = token
            self.token_counts[token] = 1
        else:
            self.token_counts[token] += 1

    def __tokenize(self, sequence: str) -> None:
        for text in sequence.split(' '):
            self.add_token(text)

    def get_index_token(self, index: int) -> str | None:
        if index > self.num_tokens:
            return
        return self.index_token[index]

    def get_token_index(self, token: str) -> int:
        if token not in self.token_index:
            self.add_token(token)
        return self.token_index[token]

    def __fit(self, sequence: str) -> np.ndarray:
        digit = list()
        for text in sequence.split(' '):
            digit.append(self.get_token_index(text))
        return np.array(digit)
    
    def tokenize(self, sequences: list) -> list:
        digit_sequences = []
        if self.tokenizer_path is not None:
            self.__load_tokenizer()
        for sequence in sequences:
            self.__tokenize(sequence=sequence)
            digit_sequences.append(self.__fit(sequence=sequence))
        if self.tokenizer_path is not None:
            self.__save_tokenizer(self.tokenizer_path)

        return digit_sequences

    def save_tokenizer(self, tokenizer_path: str) -> None:
        self.__save_tokenizer(tokenizer_path=tokenizer_path)
        self.tokenizer_path = tokenizer_path


class TextProcessor:
    def __init__(self, tokenizer_path: str = None, filters: str = '[@!*?|\/~`]', remove_stop_words: bool = False) -> None:
        self.sign_handler = SignHandler()
        self.cleanner = Cleanner(filters=filters)
        self.replacer = Replacer()
        self.remover = Remover()
        self.tokenizer = Tokenizer(tokenizer_path=tokenizer_path)
        self.remove_stop_words = remove_stop_words

    def padding_sequence(self, sequence, padding: str, maxlen: int) -> np.ndarray:
        delta = maxlen - len(sequence)
        zeros = np.zeros(delta, dtype=np.int64)

        if padding.strip().lower() == 'post':
            return np.concatenate((sequence, zeros), axis=0)
        elif padding.strip().lower() == 'pre':
            return np.concatenate((zeros, sequence), axis=0)

    def truncating_sequence(self, sequence, truncating: str, maxlen: int) -> np.ndarray:
        if truncating.strip().lower() == 'post':
            return sequence[0:maxlen]
        elif truncating.strip().lower() == 'pre':
            delta = sequence.shape[0] - maxlen
            return sequence[delta: len(sequence)]

    def pad_sequences(self, sequences: list, maxlen: int, padding: str = 'post', truncating: str = 'post') -> np.ndarray:
        result = []
        for _, sequence in enumerate(sequences):
            delta = sequence.shape[0] - maxlen
            if delta < 0:
                sequence = self.padding_sequence(sequence, padding, maxlen)
            elif delta > 0:
                sequence = self.truncating_sequence(sequence, truncating, maxlen)
            result.append(sequence)
        
        return np.array(result)

    def process(self, data: list, max_len: int = None, padding: str = 'post', truncating: str = "post", start_token: bool = False, end_token: bool = False) -> np.ndarray:
        
        data = self.cleanner.clean(data)
        # data = self.lemmaization.lemma(data)
        data = self.replacer.replace(data)
        sequences = self.sign_handler.sign(data, start_token=start_token, end_token=end_token)

        if self.remove_stop_words:
            sequences = self.remover.remove(sequences=sequences)
        sequences = self.tokenizer.tokenize(sequences=sequences)
        if max_len is not None:
            sequences = self.pad_sequences(sequences=sequences, maxlen=max_len, padding=padding, truncating=truncating)
        return sequences

    def __load_data(self, path: str) -> np.ndarray:
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def __save_data(self, data: np.ndarray, path: str, filename: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(f"{path}/{filename}", file, protocol=pickle.HIGHEST_PROTOCOL)

    def __load_tokenizer(self, path: str) -> None:
        with open(path, 'rb') as file:
            self.tokenizer = pickle.load(file)

    def save_data(self, data: np.ndarray, path: str, filename: str, overwrite: bool = True) -> None:
        if os.path.exists(path) == True:
            if overwrite == True:
                self.__save_data(data, path, filename)
            else:
                print("Not Overwrite")
        else:
            print("Not found path")

    def load_data(self, path: str) -> np.ndarray | None:
        if os.path.exists(path) == True:
            data = self.__load_data(path)
            return data
        else:
            print("Not found path")

    def loadd_tokenizer(self, path: str) -> None:
        if os.path.exists(path) == True:
            self.__load_tokenizer(path)
            print("Loaded Tokenizer")
        else:
            print("Not found path")