from typing import Dict, Iterable, List, Set

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, ListField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer


@DatasetReader.register("zero_shot_classification")
class ZeroShotClassificationReader(DatasetReader):
    """
    A dataset reader intended for use with the "emotion" and "topic" categories in the data from the
    "Benchmarking Zero-Shot Text Classification" paper, ACL 2019.

    # Parameters

    tokenizer: `Tokenizer`

        Determines how to split strings of text into tokens.

    token_indexers: `Dict[str, TokenIndexer]`

        Determines how strings get represented as tensors before being passed to the model.

    max_tokens: `int`

        If given, we'll truncate input text to this length

    label_file: `str`

        If given, this should be a file containing a list of labels, one label per line.  The topic
        dataset is given with a integers as labels in the actual data, with a separate file
        containing the mapping from those integers to labels.  For convenience, we'll map them to
        their strings here using this file, so our vocabulary works as it's intended.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        label_file: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.label_mapping: Dict[int, str] = None
        if label_file:
            self.label_mapping = {}
            for i, line in enumerate(open(label_file)):
                self.label_mapping[i] = line.strip()

    def _read(self, file_path: str) -> Iterable[Instance]:
        # Because we need to get a list of all of the labels that we can pass to the model, we go
        # through the data twice: once to read all the labels, then once to actually create the
        # instances.  We cache what we read the first time.
        instances = []
        all_labels: Set[str] = set()
        with open(file_path, "r") as lines:
            for line in lines:
                # The emotion dataset has three columns, for some reason, while the topic dataset
                # has two.  This handles both cases.
                parts = line.strip().split("\t")
                label = parts[0]
                text = parts[-1]

                if self.label_mapping:
                    label = self.label_mapping[int(label)]
                all_labels.add(label)
                instances.append((label, text))
        for label, text in instances:
            yield self.text_to_instance(text, all_labels, label)

    def text_to_instance(
        self, text: str, labels: Set[str], gold_label: str = None
    ) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)

        # We'll represent the candidate labels as a list of TextFields, because that's the easiest
        # way to share vocabularies between the text embedding and the labels, which is the whole
        # point of our zero-shot modeling right now.  It'll also be really convenient in later model
        # iterations, as we'll see.
        label_text_fields = []
        for label in labels:
            # Potential optimization: pre-tokenize the labels. But we won't bother here.
            label_tokens = self.tokenizer.tokenize(label)
            label_text_fields.append(TextField(label_tokens, self.token_indexers))
        labels_field = ListField(label_text_fields)
        fields = {"text": text_field, "labels": labels_field}
        if gold_label:
            fields["gold_label"] = LabelField(gold_label)
        return Instance(fields)
