from unittest import TestCase

from zero_shot.simple_embedding.dataset_reader import ZeroShotClassificationReader


class ZeroShotClassificationReaderTest(TestCase):
    def test_read_topic_data(self):
        reader = ZeroShotClassificationReader(label_file="test_fixtures/topic_classes.txt")
        instances = reader.read("test_fixtures/topic_dev.txt")
        assert len(instances) == 7
        assert [t.text for t in instances[0]["text"][:3]] == ["What", "makes", "friendship"]
        assert len(instances[0]["labels"]) == 6  # there are 6 unique labels in the fixture
        assert instances[0]["gold_label"].label == "Family & Relationships"
