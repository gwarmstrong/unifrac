from typing import Union, List, Tuple

import biom
import numpy as np
import skbio
from biom import load_table


# TODO change signature to be similar to
#  `skbio.diversity.beta.unweighted_unifrac`
def hotspot_pairs(table: Union[str, biom.Table],
                  tree: Union[str, skbio.TreeNode],
                  pairs: List[Tuple[str, str]],
                  method: str = 'weighted'):

    if isinstance(tree, str):
        tree = skbio.TreeNode.read(tree)
    elif not isinstance(tree, skbio.TreeNode):
        raise ValueError("Unsupported type {} for tree.".format(
            type(table)))

    if isinstance(table, str):
        table = load_table(table)
    elif not isinstance(table, biom.Table):
        raise ValueError("Unsupported")

    # TODO assert all ids in table are in tree

    tree.assign_ids()

    observation_names = table.ids('observation')

    all_results = dict()
    for sample_1, sample_2 in pairs:
        sample_1_data = table.data(sample_1)
        sample_2_data = table.data(sample_2)
        all_results[(sample_1, sample_2)] = hotspot(tree,
                                                    observation_names,
                                                    sample_1_data,
                                                    sample_2_data,
                                                    method=method)
    return all_results


# TODO change signature to be similar to
#  `skbio.diversity.beta.unweighted_unifrac`
def hotspot(tree: Union[str, skbio.TreeNode],
            names: np.ndarray,
            sample_1: np.ndarray,
            sample_2: np.ndarray,
            method='weighted') -> dict():
    # basically just takes the max of diff_abund from
    # _emd_unifrac_single_pair and returns the profile for that hotspot
    if isinstance(tree, str):
        tree = skbio.TreeNode.read(tree)
    elif not isinstance(tree, skbio.TreeNode):
        raise ValueError("Unsupported type for tree.")

    # TODO have `method` option adjust weights (could put up the stack to
    #  avoid repeating adjustment for samples)
    sample_1, sample_2 = _weight_adjuster(sample_1, sample_2, method=method)

    # makes sure each node has an id so we can refer to it later
    # ids are assigned by postorder traversal
    tree.assign_ids()

    ids = [tree.find(name).id for name in names]

    _, differential_abundances = _emd_unifrac_single_pair(tree,
                                                          ids,
                                                          sample_1,
                                                          sample_2)
    # find most extreme differential abundance
    max_abs_diff_abund = 0
    signed_max_abs_diff_abund = 0
    # TODO edge case where samples are exactly same
    for id_, diff_abund in differential_abundances:
        abs_diff_abund = abs(diff_abund)
        if abs_diff_abund > max_abs_diff_abund:
            max_abs_diff_abund = abs_diff_abund
            signed_max_abs_diff_abund = diff_abund
            max_change_node_id = id_

    max_change_node = tree.find_by_id(max_change_node_id)

    profile = _profile_hotspot(max_change_node)
    profile.update({'differential_abundance': signed_max_abs_diff_abund})
    return profile


def _weight_adjuster(sample_1: np.ndarray,
                     sample_2: np.ndarray,
                     method: str = 'weighted'):
    # TODO assumes non-negative data
    if method == 'unweighted':
        sample_1 = (sample_1 > 0).astype(float)
        sample_2 = (sample_2 > 0).astype(float)
    elif method == 'weighted':
        sample_1 = sample_1 / sample_1.sum()
        sample_2 = sample_2 / sample_2.sum()
    else:
        raise ValueError("method: '{}' not recognized.".format(method))

    return sample_1, sample_2


def _profile_hotspot(hotspot_node: skbio.TreeNode) -> dict:
    # returns dict with {'distance_to_root', 'clade_width',
    # 'maximally_divergent_tips', 'node_address'}
    clade_width, maximally_divergent_tips = hotspot_node.get_max_distance()
    output = {'distance_to_root': hotspot_node.distance(hotspot_node.root()),
              'clade_width': clade_width,
              'maximally_divergent_tips': maximally_divergent_tips,
              'node_address': hotspot_node.id}
    return output


# TODO change signature to be similar to
#  `skbio.diversity.beta.unweighted_unifrac`
def _emd_unifrac_single_pair(tree: skbio.TreeNode,
                             ids: list,  # really could be any enumerated type
                             sample_1_weights: np.ndarray,
                             sample_2_weights: np.ndarray) -> \
                             Tuple[float, dict]:
    # unweighted is present in the source
    # https://github.com/dkoslicki/EMDUnifrac/blob/master/src/EMDUnifrac.py
    # EMDUnifrac_unweighted
    # runs emd_unifrac and returns
    # returns unifrac score and differential_abundance

    # TODO assumes all nodes have id

    distance = 0
    differential_abundance = dict()

    partial_sum_array = sample_1_weights - sample_2_weights

    partial_sums = dict(zip(ids, partial_sum_array))
    for node in tree.postorder(include_self=False):
        # TODO better variable names
        node_id = node.id
        val = partial_sums.pop(node_id, 0)
        parent_id = node.parent.id
        parent_val = partial_sums.get(parent_id, 0)
        parent_val += val
        partial_sums[parent_id] = parent_val
        if val != 0:
            # TODO edge case of identical samples?
            differential_abundance[node_id] = node.length * val
        distance += node.length * abs(val)

    return distance, differential_abundance
