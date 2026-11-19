import random
import unittest

from app import main


class ClassifierBalancingTests(unittest.TestCase):
    def test_weighted_sampler_targets_minority_frequency_without_mutating_labels(self):
        labels = [0] * 85 + [1] * 15
        original = list(labels)
        train_indices = list(range(len(labels)))
        weights = main.classifier_sampling_weights(labels, train_indices, (0.7, 0.3))
        draws = random.Random(7).choices(train_indices, weights=weights, k=10000)
        sampled_minority = sum(labels[index] == 1 for index in draws)
        self.assertGreater(sampled_minority / len(draws), 0.25)
        self.assertAlmostEqual(sampled_minority / len(draws), 0.3, delta=0.025)
        self.assertEqual(labels, original)

    def test_sampler_only_sees_training_indices_and_evaluation_splits_remain_unique(self):
        labels = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
        train = [0, 1, 2, 3, 4, 5]
        validation = [6, 7]
        test = [8, 9]
        weights = main.classifier_sampling_weights(labels, train, (0.7, 0.3))
        weighted_indices = random.Random(9).choices(train, weights=weights, k=len(train))
        self.assertTrue(set(weighted_indices).issubset(train))
        self.assertTrue(set(weighted_indices).isdisjoint(validation))
        self.assertTrue(set(weighted_indices).isdisjoint(test))
        self.assertEqual(len(validation), len(set(validation)))
        self.assertEqual(len(test), len(set(test)))

    def test_metrics_report_each_class_and_highlight_minority_recall(self):
        result = main.classifier_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(result["confusionMatrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["perClass"]["1_container"]["recall"], 0.5)
        self.assertEqual(result["perClass"]["2_containers"]["recall"], 0.5)
        self.assertEqual(result["recall"], result["perClass"]["2_containers"]["recall"])

    def test_old_mobilenet_checkpoint_payload_layout_is_accepted(self):
        state = {"classifier.3.weight": "existing-weight-tensor"}
        payload = {"state_dict": state, "architecture": "mobilenet_v3_small",
                   "classes": ["1_container", "2_containers"], "imageSize": 224}
        self.assertIs(main.classifier_checkpoint_state(payload, "mobilenet_v3_small",
                                                       ["1_container", "2_containers"]), state)
        with self.assertRaises(ValueError):
            main.classifier_checkpoint_state(payload, "mobilenet_v3_large",
                                             ["1_container", "2_containers"])

    def test_distribution_reports_counts_percentages_and_imbalance_ratio(self):
        result = main.classifier_distribution([0] * 858 + [1] * 142)
        self.assertEqual(result["counts"], {"1_container": 858, "2_containers": 142})
        self.assertAlmostEqual(result["percentages"]["2_containers"], 14.2)
        self.assertAlmostEqual(result["imbalanceRatio"], 6.042, places=2)


if __name__ == "__main__":
    unittest.main()
