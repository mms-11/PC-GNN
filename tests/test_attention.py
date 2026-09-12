import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.layers import IntraAgg, IntraAggAtt


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.features = nn.Embedding.from_pretrained(torch.tensor([
            [1.0, 2.0, 1.0], [2.0, 1.0, 3.0],
            [4.0, 2.0, 1.0], [1.0, 5.0, 2.0],
        ]))
        self.aggregator = IntraAggAtt(
            self.features, 3, 6, [2, 3], 0.5,
            use_choose=False, attn_dim=2)
        with torch.no_grad():
            self.aggregator.weight.copy_(torch.eye(6))
            self.aggregator.W_q.copy_(torch.tensor([
                [0.1, 0.2, 0.0], [0.0, 0.1, 0.2]]))
            self.aggregator.W_k.copy_(torch.tensor([
                [0.2, 0.0, 0.1], [0.1, 0.2, 0.0]]))

    def arguments(self, neighbors, train_flag=False):
        return ([0, 1], torch.tensor([0, 1]), neighbors,
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                [torch.tensor([[float(node), 0.0] for node in group]).reshape(-1, 2)
                 for group in neighbors],
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                [2, 2], train_flag)

    def assert_close(self, actual, expected):
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5),
                        'Actual: {}\nExpected: {}'.format(actual, expected))

    def test_attention_matches_neighbor_only_reference(self):
        neighbors = [[1, 2], [3]]
        output, _ = self.aggregator(*self.arguments(neighbors))
        for row, group in enumerate(neighbors):
            central = self.features.weight[row]
            values = self.features(torch.tensor(group))
            query = self.aggregator.W_q.mv(central)
            logits = torch.stack([
                torch.dot(query, self.aggregator.W_k.mv(value))
                for value in values]) / math.sqrt(2)
            probabilities = torch.softmax(logits, dim=0)
            expected = (probabilities[:, None] * values).sum(dim=0)
            self.assert_close(output[row, :3], central)
            self.assert_close(output[row, 3:], expected)

    def test_uniform_attention_matches_baseline_with_choose(self):
        self.aggregator.use_choose = True
        baseline = IntraAgg(self.features, 3, 6, [2, 3], 0.5)
        with torch.no_grad():
            self.aggregator.W_q.zero_()
            baseline.weight.copy_(self.aggregator.weight)
        for train_flag in (True, False):
            with self.subTest(train_flag=train_flag):
                arguments = self.arguments([[0, 1, 2, 3], [0, 1, 2, 3]], train_flag)
                actual, distances = self.aggregator(*arguments)
                expected, baseline_distances = baseline(*self.arguments(
                    [[0, 1, 2, 3], [0, 1, 2, 3]], train_flag))
                self.assert_close(actual, expected)
                self.assertEqual(distances, baseline_distances)

    def test_query_and_key_receive_finite_nonzero_gradients(self):
        output, _ = self.aggregator(*self.arguments([[1, 2], [2, 3]]))
        output.square().sum().backward()
        for parameter in (self.aggregator.W_q, self.aggregator.W_k):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_empty_neighbors_preserve_center_and_finite_gradients(self):
        for neighbors in ([[], []], [[], [2, 3]]):
            with self.subTest(neighbors=neighbors):
                self.aggregator.zero_grad()
                output, _ = self.aggregator(*self.arguments(neighbors))
                self.assertTrue(torch.isfinite(output).all().item())
                self.assert_close(output[:, :3], self.features.weight[:2])
                self.assert_close(output[0, 3:], torch.zeros(3))
                if not neighbors[1]:
                    self.assert_close(output[1, 3:], torch.zeros(3))
                output.square().sum().backward()
                for parameter in self.aggregator.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_no_choose_bypasses_both_selectors(self):
        with patch('src.layers.choose_step_neighs') as train_choose, \
                patch('src.layers.choose_step_test') as test_choose:
            for train_flag in (True, False):
                self.aggregator(*self.arguments([[1, 2], [3]], train_flag))
            train_choose.assert_not_called()
            test_choose.assert_not_called()

    def test_attention_uses_only_choose_output(self):
        self.aggregator.use_choose = True
        for train_flag, selector in ((True, 'choose_step_neighs'),
                                     (False, 'choose_step_test')):
            with self.subTest(train_flag=train_flag):
                with patch('src.layers.' + selector,
                           return_value=([{2}, {3}], [[0.1], [0.2]])) as choose:
                    output, distances = self.aggregator(*self.arguments(
                        [[0, 1, 2], [0, 1, 3]], train_flag))
                    choose.assert_called_once()
                    self.assertEqual(distances, [[0.1], [0.2]])
                    self.assert_close(output[:, 3:], self.features.weight[2:4])

    def test_dynamic_feature_dimensions(self):
        for dimension in (1, 25, 32, 100):
            with self.subTest(dimension=dimension):
                features = nn.Embedding(4, dimension)
                aggregator = IntraAggAtt(features, dimension, 5, [2, 3], 0.5,
                                         use_choose=False)
                self.assertEqual(aggregator.attn_dim, max(dimension // 2, 1))
                self.assertEqual(tuple(aggregator.weight.shape), (2 * dimension, 5))
                output, _ = aggregator(*self.arguments([[1, 2], [3]]))
                self.assertEqual(tuple(output.shape), (2, 5))
                self.assertTrue(torch.isfinite(output).all().item())

    def test_invalid_attention_dimensions(self):
        for dimension in (0, -1, 1.5, True):
            with self.subTest(dimension=dimension):
                with self.assertRaises(ValueError):
                    IntraAggAtt(self.features, 3, 6, [], 0.5, attn_dim=dimension)


if __name__ == '__main__':
    unittest.main()