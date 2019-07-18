from typing import Union, List, Tuple, Dict
import biom
import numpy as np
import skbio
from biom import load_table

available_metrics = {'weighted_unifrac', 'unweighted_unifrac'}


def hotspot_pairs(table: Union[str, biom.Table],
                  tree: Union[str, skbio.TreeNode],
                  pairs: List[Tuple[str, str]],
                  metric: str = 'weighted_unifrac'):
    """

    Parameters
    ----------
    table
        An instance or filepath to a BIOM 2.1 formatted table (HDF5)
    tree
        A skbio.TreeNode or filepath to a Newick formatted tree
    pairs
        A list of pairs of Sample ID's in the given `table`
        corresponding to pairs that should be compared to generate
        hotspots
    metric
        The pairwise distance function to apply.
        Options: `weighted_unifrac` and `unweighted_unifrac`

    Returns
    -------
        Dictionary containing a dictionary of node statistics for the
        `hotspot` of reach pair in `pairs`

    Raises
    ------
    # TODO

    Notes
    -----
    # TODO

    References
    ----------
    # TODO

    """

    if isinstance(tree, str):
        tree = skbio.TreeNode.read(tree)
    elif not isinstance(tree, skbio.TreeNode):
        raise ValueError("Unsupported type {} for tree.".format(
            type(tree)))

    if isinstance(table, str):
        table = load_table(table)
    elif not isinstance(table, biom.Table):
        raise ValueError("Unsupported type {} for table.".format(
            type(table)))

    if metric not in available_metrics:
        raise ValueError("Unsupported metric '{}'".format(metric))

    # TODO assert all ids in table are in tree, might be able to reuse
    #  validation functions from skbio

    tree.assign_ids()

    observation_names = table.ids('observation')

    all_results = dict()
    for sample_1, sample_2 in pairs:
        sample_1_data = table.data(sample_1)
        sample_2_data = table.data(sample_2)
        all_results[(sample_1, sample_2)] = hotspot(sample_1_data,
                                                    sample_2_data,
                                                    observation_names,
                                                    tree,
                                                    metric=metric)
    return all_results


def hotspot(u_counts: np.array,
            v_counts: np.array,
            otu_ids: Union[List, np.array],
            tree: Union[str, skbio.TreeNode],
            metric='weighted_unifrac') -> Dict:
    """

    Parameters
    ----------
    u_counts, v_counts
        Vectors of counts/relative abundances of OTUs for two samples. Must be
        equal length.
    otu_ids
        Vector of OTU ids corresponding to tip names in tree. Must be the
        same length as u_counts and v_counts.
    tree
        Tree relating the OTUs in otu_ids. The set of tip names in the
        tree can be a superset of otu_ids, but not a subset.
    metric
        The pairwise distance function to apply. Options: `weighted_unifrac`
        and `unweighted_unifrac`

    Returns
    -------
        Dictionary of statistics on the node identified as contributing the
        most to the distance specified by `metric`

    Raises
    ------
    # TODO

    Notes
    -----
    # TODO

    References
    ----------
    # TODO

    """
    # basically just takes the max of diff_abund from
    # _emd_unifrac_single_pair and returns the profile for that hotspot
    # TODO allow stringIO or file object?
    if isinstance(tree, str):
        tree = skbio.TreeNode.read(tree)
    elif not isinstance(tree, skbio.TreeNode):
        raise ValueError("Unsupported type {} for tree.".format(type(tree)))
    tree = tree.shear(otu_ids)
    max_change_node = _calculate_hotspot(u_counts, v_counts, otu_ids, tree,
                                         metric)
    profile = _profile_tree_node(max_change_node)
    return profile


# TODO with EMDUnifrac there may be some concern over uniqueness of
#  hotspots, but I am not sure how this will play out on real data
def _calculate_hotspot(u_counts: np.array,
                       v_counts: np.array,
                       otu_ids: Union[List, np.array],
                       tree: skbio.TreeNode,
                       metric='weighted_unifrac') -> skbio.TreeNode:
    # general idea with EMD unifrac: should be able to adjust weights for
    # different unifrac methods
    u_counts, v_counts = _weight_adjuster(u_counts,
                                          v_counts,
                                          tree,
                                          method=metric)

    # makes sure each node has an id so we can refer to it later
    # ids are assigned by postorder traversal
    tree.assign_ids()

    ids = [tree.find(name).id for name in otu_ids]

    _, differential_abundances = _emd_unifrac_single_pair(tree,
                                                          ids,
                                                          u_counts,
                                                          v_counts)
    # find most extreme differential abundance
    max_abs_diff_abund = -1
    # max_change_node_id = tree.find('root')
    # TODO edge case where samples are exactly same
    for id_, diff_abund in differential_abundances.items():
        abs_diff_abund = abs(diff_abund)
        if abs_diff_abund > max_abs_diff_abund:
            max_abs_diff_abund = abs_diff_abund
            max_change_node_id = id_

    max_change_node = tree.find_by_id(max_change_node_id)

    return max_change_node


def _weight_adjuster(sample_1: np.array,
                     sample_2: np.array,
                     tree: skbio.TreeNode = None,
                     method: str = 'weighted_unifrac'):
    # TODO assumes non-negative data
    if method == 'unweighted_unifrac':
        if tree is None:
            raise ValueError('Unweighted weight adjustment requires a tree.')
        total_branch_length = tree.descending_branch_length()
        sample_1 = (sample_1 > 0).astype(float) / total_branch_length
        sample_2 = (sample_2 > 0).astype(float) / total_branch_length
    elif method == 'weighted_unifrac':
        sample_1 = sample_1 / sample_1.sum()
        sample_2 = sample_2 / sample_2.sum()
    else:
        raise ValueError("method: '{}' not recognized.".format(method))

    return sample_1, sample_2


def _profile_tree_node(node: skbio.TreeNode) -> dict:
    # returns dict with {'distance_to_root', 'clade_width',
    # 'maximally_divergent_tips', 'node_address'}
    clade_width, maximally_divergent_tips = node.get_max_distance()
    output = {'distance_to_root': node.distance(node.root()),
              'clade_width': clade_width,
              'maximally_divergent_tips': maximally_divergent_tips,
              'node_address': node.id}
    return output


# TODO potentially revist signature
def _emd_unifrac_single_pair(tree: skbio.TreeNode,
                             ids: list,  # really could be any enumerated type
                             sample_1_weights: np.array,
                             sample_2_weights: np.array) -> \
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
