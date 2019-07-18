import unittest
from unittest import mock
from io import StringIO
from unittest import TestCase
import pkg_resources

import numpy as np
from skbio import TreeNode
from skbio.diversity import beta
from biom import load_table

from unifrac.longitudinal import (_emd_unifrac_single_pair,
                                  _weight_adjuster,
                                  hotspot,
                                  hotspot_pairs)


class TestHotspotPairs(unittest.TestCase):

    package = 'unifrac.tests'

    def setUp(self):
        self.tree_path = self.get_data_path('t1.newick')
        self.table_path = self.get_data_path('e1.biom')

    def get_data_path(self, filename):
        # adapted from qiime2.plugin.testing.TestPluginBase
        return pkg_resources.resource_filename(self.package,
                                               'data/%s' % filename)

    def test_hotspot_pairs_paths(self):
        pairs = [('A', 'B'), ('B', 'C')]
        hotspot_dict = hotspot_pairs(self.table_path,
                                     self.tree_path,
                                     pairs,
                                     metric='unweighted_unifrac')
        a_b_node_id = 7
        b_c_node_id = 4
        self.assertEqual(a_b_node_id, hotspot_dict[('A', 'B')]['node_address'])
        self.assertEqual(b_c_node_id, hotspot_dict[('B', 'C')]['node_address'])

    def test_hotspot_pairs_objects(self):
        pairs = [('A', 'B'), ('B', 'C')]
        table = load_table(self.table_path)
        tree = TreeNode.read(self.tree_path)
        hotspot_dict = hotspot_pairs(table,
                                     tree,
                                     pairs,
                                     metric='unweighted_unifrac')
        a_b_node_id = 7
        b_c_node_id = 4
        self.assertEqual(a_b_node_id, hotspot_dict[('A', 'B')]['node_address'])
        self.assertEqual(b_c_node_id, hotspot_dict[('B', 'C')]['node_address'])

    def test_hotspot_pairs_unsupported_tree_type(self):
        pairs = [('A', 'B'), ('B', 'C')]
        with self.assertRaisesRegex(ValueError, r"Unsupported.*tree"):
            hotspot_pairs(self.table_path,
                          [],
                          pairs,
                          metric='unweighted_unifrac')

    def test_hotspot_pairs_unsupported_table_type(self):
        pairs = [('A', 'B'), ('B', 'C')]
        with self.assertRaisesRegex(ValueError, r"Unsupported.*table"):
            hotspot_pairs([],
                          self.tree_path,
                          pairs,
                          metric='unweighted_unifrac')

    def test_hotspot_pairs_unsupported_table_metric(self):
        pairs = [('A', 'B'), ('B', 'C')]
        with self.assertRaisesRegex(ValueError, r"Unsupported metric"):
            hotspot_pairs(self.table_path,
                          self.tree_path,
                          pairs,
                          metric='not-a-real-metric')


class TestHotspot(unittest.TestCase):

    package = 'unifrac.tests'

    def setUp(self):
        self.tree_str = '(((1:0.3, 2:0.3)6:0.5,(3:0.1,4:0.1)7:0.1)8:0.1,' \
                    '5:0.1)root;'
        tree = TreeNode.read(StringIO(self.tree_str))
        tree.assign_ids()
        self.tree = tree
        self.otu_ids = ['1', '2', '3', '4', '5']
        self.u_counts = [0, 0, 1, 1, 2]
        self.v_counts = [1, 1, 1, 1, 0]

    def get_data_path(self, filename):
        # adapted from qiime2.plugin.testing.TestPluginBase
        return pkg_resources.resource_filename(self.package,
                                               'data/%s' % filename)

    def test_hotspot_gets_profile(self):
        with mock.patch('unifrac.longitudinal._calculate_hotspot') as \
                mocked_hotspot:
            mocked_return_node = self.tree
            mocked_hotspot.return_value = mocked_return_node
            hotspot_profile = hotspot(self.u_counts,
                                      self.v_counts,
                                      self.otu_ids,
                                      self.tree,
                                      metric='weighted_unifrac')

        observed_node = self.tree.find_by_id(hotspot_profile['node_address'])
        self.assertCountEqual(mocked_return_node, observed_node)

    def test_hotspot_gets_profile_table_smaller_than_tree(self):
        with mock.patch('unifrac.longitudinal._calculate_hotspot') as \
                mocked_hotspot:
            mocked_return_node = self.tree
            mocked_hotspot.return_value = mocked_return_node
            u_counts = self.u_counts.copy()
            v_counts = self.v_counts.copy()
            otu_ids = self.otu_ids.copy()
            u_counts.pop(), v_counts.pop(), otu_ids.pop()
            hotspot_profile = hotspot(self.u_counts,
                                      self.v_counts,
                                      self.otu_ids,
                                      self.tree,
                                      metric='weighted_unifrac')

        observed_node = self.tree.find_by_id(hotspot_profile['node_address'])
        self.assertCountEqual(mocked_return_node, observed_node)

    def test_hotspot_gets_profile_works_on_path(self):
        with mock.patch('unifrac.longitudinal._calculate_hotspot') as \
                mocked_hotspot:
            mocked_return_node = self.tree
            t1 = self.get_data_path('t1.newick')
            mocked_hotspot.return_value = mocked_return_node
            hotspot_profile = hotspot(self.u_counts,
                                      self.v_counts,
                                      ['a', 'b', 'c', 'd', 'e'],
                                      t1,
                                      metric='weighted_unifrac')

        observed_node = self.tree.find_by_id(hotspot_profile['node_address'])
        self.assertCountEqual(mocked_return_node, observed_node)

    def test_hotspot_unsupported_tree_type(self):
        with self.assertRaisesRegex(ValueError, r'Unsupported type .* for '
                                                r'tree'):
            hotspot(self.u_counts, self.v_counts, self.otu_ids,
                    ['list', 'of', 'things'])


class TestEMDUnifrac(unittest.TestCase):
    package = 'unifrac.tests'

    # see tests in
    # https://github.com/dkoslicki/EMDUnifrac/blob/master/src/EMDUnifrac.py
    def test_weighted_emd_unifrac_from_paper(self):
        tree_str = '((1:0.1,2:0.1)5:0.2,(3:0.1,4:0.1)6:0.2)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        names = ['1', '2', '3', '4', '6']
        ids = [tree.find(name).id for name in names]
        sample_1 = np.array([0, 1 / 2, 1 / 2, 0, 0])
        sample_2 = np.array([1 / 3, 0, 0, 1 / 3, 1 / 3])

        # expected = beta.weighted_unifrac(sample_1, sample_2, names, tree)
        distance, diff_abund = _emd_unifrac_single_pair(tree,
                                                        ids,
                                                        sample_1,
                                                        sample_2)
        id_mapping = {node.id: node.name for
                      node in tree.postorder()}
        diff_abund_names = {id_mapping[id_]: abund for id_, abund in
                            diff_abund.items()}

        print(diff_abund_names)

        # answer from paper
        expected = 0.2333333

        self.assertAlmostEqual(expected, distance)

    def test_weighted_emd_unifrac_root_diff(self):
        # TODO check with skbio
        tree_str = '((1:0.2,2:0.1)5:0.3,(3:0.1,4:0.2)6:0.3)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        names = ['1', '2', '3', '4']
        ids = [tree.find(name).id for name in names]
        sample_1 = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])
        sample_2 = np.array([1 / 2, 1 / 2, 0, 0])
        distance, diff_abund = _emd_unifrac_single_pair(tree,
                                                        ids,
                                                        sample_1,
                                                        sample_2)

        id_mapping = {node.id: node.name for
                      node in tree.postorder()}
        print(id_mapping)
        diff_abund_names = {id_mapping[id_]: abund for id_, abund in
                            diff_abund.items()}

        print(diff_abund_names)
        # print(diff_abund)
        self.assertAlmostEqual(0.45, distance)

    def test_weighted_emd_unifrac_bigger_example(self):
        tree_str = '(((1:0.3, 2:0.3)6:0.5,(3:0.1,4:0.1)7:0.1)8:0.1,5:0.1)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        names = ['1', '2', '3', '4', '5']
        ids = [tree.find(name).id for name in names]
        sample_1 = np.array([0, 0, 1 / 4, 1 / 4, 1 / 2])
        sample_2 = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4, 0])
        distance, diff_abund = _emd_unifrac_single_pair(tree,
                                                        ids,
                                                        sample_1,
                                                        sample_2)

        id_mapping = {node.id: node.name for
                      node in tree.postorder()}
        print(id_mapping)
        diff_abund_names = {id_mapping[id_]: abund for id_, abund in
                            diff_abund.items()}

        print(diff_abund_names)
        # print(diff_abund)
        # multiply by 4 to get counts
        expected = beta.weighted_unifrac(4 * sample_1, 4 * sample_2, names,
                                         tree)
        self.assertAlmostEqual(expected, distance)

    def test_unweighted_emd_unifrac_from_author_implementation(self):
        # tree_str = '((B:0.1,C:0.2)A:0.3)root;' # author's tree
        tree_str = '(B:0.1,C:0.2)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        names = ['B', 'C']
        ids = [tree.find(name).id for name in names]
        sample_1 = np.array([1, 1]) / tree.descending_branch_length()
        sample_2 = np.array([1, 0]) / tree.descending_branch_length()
        expected = beta.unweighted_unifrac(sample_1, sample_2, names, tree)
        distance, diff_abund = _emd_unifrac_single_pair(tree,
                                                        ids,
                                                        sample_1,
                                                        sample_2)

        # distance = distance / tree.descending_branch_length()

        id_mapping = {tree.find_by_id(id_).id: tree.find_by_id(id_).name for
                      id_ in ids}
        diff_abund_names = {id_mapping[id_]: abund for id_, abund in
                            diff_abund.items()}
        print(diff_abund_names)
        self.assertAlmostEqual(expected, distance)

    def test_unweighted_emd_unifrac_from_paper(self):
        tree_str = '((1:0.1,2:0.1)5:0.2,(3:0.1,4:0.1)6:0.2)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        names = ['1', '2', '3', '4']
        ids = [tree.find(name).id for name in names]
        sample_1 = np.array([0, 1, 1, 0]) / tree.descending_branch_length()
        sample_2 = np.array([1, 0, 0, 1]) / tree.descending_branch_length()
        # get expected from skbio
        expected = beta.unweighted_unifrac(sample_1, sample_2, names, tree)
        distance, diff_abund = _emd_unifrac_single_pair(tree,
                                                        ids,
                                                        sample_1,
                                                        sample_2)
        # distance = distance / tree.descending_branch_length()

        print(diff_abund)
        print({tree.find_by_id(id_).name: tree.find_by_id(id_) for id_ in ids})
        print('expected: {}'.format(expected))
        self.assertAlmostEqual(expected, distance)
        # TODO check diff_abund


class TestWeightAdjuster(TestCase):

    package = 'unifrac.tests'

    def setUp(self):
        tree_str = '((1:0.2,2:0.1)5:0.3,(3:0.1,4:0.2)6:0.3)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        self.tree = tree
        names = ['1', '2', '3', '4']
        self.ids = [tree.find(name).id for name in names]
        self.sample_1 = np.array([1, 1, 1, 1])
        self.sample_2 = np.array([1, 1, 0, 0])

    def test_weighted_adjuster(self):
        expected_1 = np.array([1/4, 1/4, 1/4, 1/4])
        expected_2 = np.array([1/2, 1/2, 0, 0])
        adjusted_s1, adjusted_s2 = _weight_adjuster(self.sample_1,
                                                    self.sample_2,
                                                    method='weighted_unifrac')

        self.assertTrue(np.allclose(expected_1, adjusted_s1))
        self.assertTrue(np.allclose(expected_2, adjusted_s2))

    def test_unweighted_adjuster(self):
        length = 1.2
        expected_1 = np.array([1/length, 1/length, 1/length, 1/length])
        expected_2 = np.array([1/length, 1/length, 0, 0])

        adjusted_s1, adjusted_s2 = _weight_adjuster(self.sample_1,
                                                    self.sample_2,
                                                    tree=self.tree,
                                                    method='unweighted_unifrac')

        self.assertTrue(np.allclose(expected_1, adjusted_s1))
        self.assertTrue(np.allclose(expected_2, adjusted_s2))


if __name__ == "__main__":
    unittest.main()


