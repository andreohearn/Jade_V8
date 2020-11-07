# create a tokenizer class
import json
import os
import io

class Tokenizer:
    def __init__(self, location):
        self.save_location = location
        if os.path.exists(self.save_location):
            with io.open(self.save_location, mode="r", encoding="utf-8") as f:
                saved_tokenizer = json.load(f)
            self.unk_token = int(saved_tokenizer["unk"])
            self.start_token = int(saved_tokenizer["start"])
            self.end_token = int(saved_tokenizer["end"])
            self.encode_token_dict = saved_tokenizer["encode"]
            self.decode_token_dict = saved_tokenizer["decode"]
            self.generated = True
            print("Loaded existing tokenizer.")
        else:
            print("Generate a new tokenizer with Tokenizer.generate(words)")

    def generate(self, words):
        unique_words = list(dict.fromkeys([str(i).strip() for i in words]))
        print(f"Reduced repeat characters from original {len(words)} to {len(unique_words)}")
        self.unk_token = len(unique_words)+1
        # there's probally a better way to do encoding and decoding with one dictionary,
        # but it isn't nessesary to reduce the tokenizer size untless the dataset is very large
        self.encode_token_dict = {str(unique_words[i]): str(i+1) for i in range(0, len(unique_words))}
        self.decode_token_dict = {str(i+1): str(unique_words[i]) for i in range(0, len(unique_words))}
        self.start_token = int(self.encode_token_dict["|aigenerationstart|"])
        self.end_token = int(self.encode_token_dict["|endofgeneration|"])
        with io.open(self.save_location, mode="w", encoding="utf-8") as f:
            json.dump({"unk": self.unk_token, "start": self.start_token, "end": self.end_token,
                    "encode": self.encode_token_dict, "decode": self.decode_token_dict}, f)
        self.generated = True

    def encode(self, sequence, debug=False):
        self.end_sequence = []
        self.unks = []
        if self.generated:
            for part in sequence:
                try:
                    self.end_sequence.append(self.encode_token_dict[str(part)])
                except:
                    self.unks.append(str(part))
                    self.end_sequence.append(self.unk_token)
            self.end_sequence = [int(i) for i in self.end_sequence]
            if debug == True:
                print(f"{len(self.unks)} unknowns, {str(self.unks)}")
            return self.end_sequence
        else:
            raise "NotGeneratedError"

    def decode(self, sequence):
        self.end_sequence = []
        if self.generated:
            for part in sequence:
                try:
                    self.end_sequence.append(self.decode_token_dict[str(part)])
                except:
                    self.end_sequence.append("<unk>")
            self.end_sequence = " ".join(self.end_sequence)
            return self.end_sequence
        else:
            raise "NotGeneratedError"