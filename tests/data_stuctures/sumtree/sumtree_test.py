import unittest

from collections import Counter
from data_structures.sumtree.sumtree import SumTree


class SumTreeTest(unittest.TestCase):

    @staticmethod
    def _create_and_initialize_sumtree() -> SumTree:
        sumtree = SumTree(capacity=10)
        for idx in range(1, 11):
            sumtree.add(idx, chr(96 + idx))
        return sumtree

    def test_adding_items_is_correct(self):
        sumtree = SumTreeTest._create_and_initialize_sumtree()
        for idx in range(97, 97 + 10):
            self.assertIn(chr(idx), sumtree.get_item_table())

    def test_sumtree_is_circular_buffer(self):
        sumtree = SumTree(capacity=10)
        for idx in range(1, 21):
            sumtree.add(idx, chr(96 + idx))

        for idx in range(97, 97 + 10):
            self.assertNotIn(chr(idx), sumtree.get_item_table())
        for idx in range(97 + 10, 97 + 20):
            self.assertIn(chr(idx), sumtree.get_item_table())

    def test_probability_space_is_sum_of_probabilities(self):
        sumtree = SumTreeTest._create_and_initialize_sumtree()
        leafs_values = sumtree.get_tree_table()[-10:]
        self.assertEqual(sumtree.get_size_of_probability_space(), sum(leafs_values))

    def test_can_sample_all_items(self):
        sumtree = SumTreeTest._create_and_initialize_sumtree()

        items = set()
        for key in range(sumtree.get_size_of_probability_space()):
            items.add(sumtree.get(key))

        for item in sumtree.get_item_table():
            self.assertIn(item, items)

    def test_probability_distribution_is_correct(self):
        items_with_probability = {'a': 3, 'b': 3, 'c': 1, 'd': 2, 'e': 10}
        probability_space = sum(items_with_probability.values())

        sumtree = SumTree(capacity=len(items_with_probability))
        for item, key in items_with_probability.items():
            sumtree.add(key, item)

        counter = Counter()
        n_samples = 1000
        for _ in range(n_samples):
            sample = sumtree.sample()
            counter[sample] += 1

        for item in sumtree.get_item_table():
            correct_probability = items_with_probability[item]/probability_space
            sumtree_probability = counter[item]/n_samples
            self.assertAlmostEqual(correct_probability, sumtree_probability, delta=0.1)


if __name__ == '__main__':
    unittest.main()
