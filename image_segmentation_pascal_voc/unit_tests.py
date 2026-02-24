import pytest


from image_segmentation_pascal_voc.data import GoldPascalVOC2012Segmentation

GoldPascalVOC2012Segmentation(".data", override=True)


class TestGoldVOCSegmentation:
    def test_simple_usage(self):
        dataset = GoldPascalVOC2012Segmentation(".data")
        assert len(dataset) > 0
        # Pascal VOC 2012 train set has 1464 images
        assert len(dataset) == 1464

    def test_with_remove_ratio(self):
        remove_ratio = 0.2
        first_dataset = GoldPascalVOC2012Segmentation(
            ".data", remove_ratio=remove_ratio
        )
        expected_length = int(1464 * (1 - remove_ratio))
        assert len(first_dataset) == expected_length

        second_dataset = GoldPascalVOC2012Segmentation(
            ".data", remove_ratio=remove_ratio
        )
        assert len(first_dataset) == len(second_dataset)

    def test_with_duplicate_failure(self):
        duplicate_table_path = "unit_test_voc_description_table_failure"
        drop_duplicate_table = True
        to_duplicate_clusters = 3
        cluster_count = 2
        duplicate_per_sample = None
        with pytest.raises(ValueError):
            GoldPascalVOC2012Segmentation(
                ".data",
                remove_ratio=0.98,
                duplicate_table_path=duplicate_table_path,
                drop_duplicate_table=drop_duplicate_table,
                to_duplicate_clusters=to_duplicate_clusters,
                cluster_count=cluster_count,
                duplicate_per_sample=duplicate_per_sample,
            )

    def test_with_duplicate(self):
        duplicate_table_path = "unit_test_voc_description_table"
        drop_duplicate_table = True
        to_duplicate_clusters = 2
        cluster_count = 2
        duplicate_per_sample = 2

        first_dataset = GoldPascalVOC2012Segmentation(
            ".data",
            remove_ratio=0.98,
            duplicate_table_path=duplicate_table_path,
            drop_duplicate_table=drop_duplicate_table,
            to_duplicate_clusters=to_duplicate_clusters,
            cluster_count=cluster_count,
            duplicate_per_sample=duplicate_per_sample,
        )

        second_dataset = GoldPascalVOC2012Segmentation(
            ".data",
            remove_ratio=0.98,
            duplicate_table_path=duplicate_table_path,
            drop_duplicate_table=drop_duplicate_table,
            to_duplicate_clusters=to_duplicate_clusters,
            cluster_count=cluster_count,
            duplicate_per_sample=duplicate_per_sample,
        )

        assert len(first_dataset) == len(second_dataset)
