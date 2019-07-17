import unittest
from io import StringIO

import numpy as np
from skbio import TreeNode
from skbio.diversity import beta

from unifrac.longitudinal import _emd_unifrac_single_pair


class EMDUnifracTests(unittest.TestCase):
    package = 'unifrac.tests'

    # see tests in
    # https://github.com/dkoslicki/EMDUnifrac/blob/master/src/EMDUnifrac.py
    def test_weighted_emd_unifrac_from_paper(self):
        tree_str = '((1:0.1,2:0.1)5:0.2,(3:0.1,4:0.1)6:0.2)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        names = ['1', '2', '3', '4', '6']
        ids = [tree.find(name).id for name in names]
        sample_1 = np.array([0, 1/2, 1/2, 0, 0])
        sample_2 = np.array([1/3, 0, 0, 1/3, 1/3])

        # expected = beta.weighted_unifrac(sample_1, sample_2, names, tree)
        distance, diff_abund = _emd_unifrac_single_pair(tree,
                                                        ids,
                                                        sample_1,
                                                        sample_2)
        # answer from paper
        expected = 0.2333333

        self.assertAlmostEqual(expected, distance)
        # TODO check diff_abund

    def test_weighted_emd_unifrac_root_diff(self):
        tree_str = '((1:0.2,2:0.1)5:0.3,(3:0.1,4:0.2)6:0.3)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        names = ['1', '2', '3', '4']
        ids = [tree.find(name).id for name in names]
        sample_1 = np.array([1/4, 1/4, 1/4, 1/4])
        sample_2 = np.array([1/2, 1/2, 0, 0])
        distance, diff_abund = _emd_unifrac_single_pair(tree,
                                                        ids,
                                                        sample_1,
                                                        sample_2)

        self.assertAlmostEqual(0.45, distance)

    def test_unweighted_emd_unifrac_from_author_implementation(self):
        # tree_str = '((B:0.1,C:0.2)A:0.3)root;' # author's tree
        tree_str = '(B:0.1,C:0.2)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        names = ['B', 'C']
        ids = [tree.find(name).id for name in names]
        sample_1 = np.array([1, 1])
        sample_2 = np.array([1, 0])
        expected = beta.unweighted_unifrac(sample_1, sample_2, names, tree)
        distance, diff_abund = _emd_unifrac_single_pair(tree,
                                                        ids,
                                                        sample_1,
                                                        sample_2)

        distance = distance / tree.descending_branch_length()

        print('expected: {}'.format(expected))
        self.assertAlmostEqual(expected, distance)

    def test_unweighted_emd_unifrac_from_paper(self):
        tree_str = '((1:0.1,2:0.1)5:0.2,(3:0.1,4:0.1)6:0.2)root;'
        tree = TreeNode.read(StringIO(tree_str))
        tree.assign_ids()
        names = ['1', '2', '3', '4']
        ids = [tree.find(name).id for name in names]
        sample_1 = np.array([0, 1, 1, 0])
        sample_2 = np.array([1, 0, 0, 1])
        # get expected from skbio
        expected = beta.unweighted_unifrac(sample_1, sample_2, names, tree)
        distance, diff_abund = _emd_unifrac_single_pair(tree,
                                                        ids,
                                                        sample_1,
                                                        sample_2)
        distance = distance / tree.descending_branch_length()

        print(diff_abund)
        print({id_: tree.find_by_id(id_) for id_ in ids})
        print('expected: {}'.format(expected))
        self.assertAlmostEqual(expected, distance)
        # TODO check diff_abund


if __name__ == "__main__":
    unittest.main()
