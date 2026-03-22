from text_classification_imdb.data import IMDbDataset


class TestIMDbDataset:
    def test_simple_usage(self):
        dataset = IMDbDataset(split="train", count=100)
        assert len(dataset) == 100
        input_ids, attention_mask, label, idx = dataset[0]
        assert input_ids.shape[0] == 256
        assert attention_mask.shape[0] == 256
        assert label in (0, 1)
        assert idx == 0

    def test_targets_property(self):
        dataset = IMDbDataset(split="train", count=50)
        targets = dataset.targets
        assert len(targets) == 50
        assert all(t in (0, 1) for t in targets)

    def test_test_split(self):
        dataset = IMDbDataset(split="test", count=20)
        assert len(dataset) == 20
        input_ids, attention_mask, label, idx = dataset[5]
        assert input_ids.shape[0] == 256
        assert attention_mask.shape[0] == 256

    def test_max_length(self):
        dataset = IMDbDataset(split="train", max_length=64, count=10)
        input_ids, attention_mask, label, idx = dataset[0]
        assert input_ids.shape[0] == 64
        assert attention_mask.shape[0] == 64

    def test_vocab_size(self):
        dataset = IMDbDataset(split="train", count=5)
        # bert-base-uncased has 30522 tokens
        assert dataset.vocab_size == 30522

    def test_index_returned(self):
        dataset = IMDbDataset(split="train", count=10)
        for i in range(len(dataset)):
            _, _, _, idx = dataset[i]
            assert idx == i
