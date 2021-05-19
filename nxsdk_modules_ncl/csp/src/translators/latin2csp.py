###############################################################
# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2018-2021 Intel Corporation.

# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.

# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
###############################################################

import numpy as np

class translateSudoku:
    """Sudoku translator to generate the list of constraints as input for CspNxNet.

    args:
        9X9 array of values between 1 and 9.
    """

    def __init__(self, puzzle=[], is_latin=False):
        """Initialize the sudoku2csp class.

        args:
            puzzle: a python nxn array representing a Sudoku puzzle, blank cells are given with the digit 0.
        """
        elements = []
        dimension = len(puzzle)
        box = int(dimension ** (1. / 2.))
        for num_1 in range(dimension):
            for digit2 in range(dimension):
                elements.append([digit2,dimension-1-num_1])

        conflicts = []
        # row and column restrictions.
        for elm_1, pos_1 in enumerate(elements):
            for elm_2, pos_2 in enumerate(elements):
                if (pos_2[0] == pos_1[0] or pos_2[1] == pos_1[1]) and elm_2 > elm_1:
                    conflicts.append([elm_1, elm_2])

        if not is_latin:
            # boxe constraints
            for elm_1, pos_1 in enumerate(elements):
                for elm_2, pos_2 in enumerate(elements):
                    if (pos_2[0] // box == pos_1[0] // box and pos_2[1] // box == pos_1[1] // box) and (
                            pos_2[0] != pos_1[0] and pos_2[1] != pos_1[1]) and elm_2 > elm_1:
                        conflicts.append([elm_1, elm_2])

        cons = []
        for conflict in conflicts:
            cons.append((conflict[0], conflict[1]))
        cues = None
        numbers = []
        for num_1 in puzzle:
            for k in num_1:
                numbers.append(k)
        cues = []
        for num_1, cue in enumerate(numbers):
            if cue != 0:
                cues.append((num_1, cue - 1))
        self.variables = elements
        self.constraints = cons
        self.number_of_variables = len(elements)
        self.number_of_constraints = len(conflicts)
        self.states_per_element = dimension
        self.cues = cues

    @property
    def var_num(self):
        return self.number_of_variables

    @property
    def dom_num(self):
        return self.states_per_element


def check_puzzle(puzzle, size=9, is_latin=False, print_statements=False):
    """Verify the satisfiability state of a Sudoku puzzle.

    :param 2D numpy array puzzle: Sudoku puzzle to be verified.
    :param int size: size of one side of the puzzle, the puzzle itself is cosidered to be size x size.
    :param is_latin: whether the puzzle is a latin square, else is assumed to be a Sudoku, in which case subgrids exist.
    :return: dictionary of violations found. Keys are 'row', 'column' and 'block' each holding as value a list of
        violations with digits from 0 to size-1.
    :rtype: dict of lists of ints.
    """
    # extract rows, columns and blocks TODO get size from puzzle so that this user input is not needed, refactor usages.
    rows, columns = puzzle, puzzle.T
    block_size = 1 if is_latin else int(np.sqrt(size))
    # return block as 1D array, if 2D is desired, change reshape(-1, block_size*block_size) by reshape(-1,
    # block_size,block_size)
    blocks = puzzle.reshape(puzzle.shape[0] // block_size, block_size, puzzle.
                            shape[1] // block_size, block_size).swapaxes(1, 2).reshape(-1, block_size * block_size)
    # Check satisfiability
    names = ['row', 'column', 'block']
    valid = True
    violations = {'row':[], 'column':[],'block':[]}
    for idxs, segment in enumerate([rows, columns, blocks]):
        for idxa, array in enumerate(segment):
            # there should be 'size' number of unique digits in array if the puzzle is solved
            if not np.unique(array).shape == array.shape:
                if print_statements:
                    print("%s %d does not satisfy all constraints" % (names[idxs], idxa + 1))
                violations[names[idxs]].append(idxa)
                # violations.append((names[idxs], idxa + 1))
                valid = False
    if valid:
        if print_statements:
            print('the puzzle satisfies all constraints')
    else:
        if print_statements:
            print('the puzzle does not satisfy all constraints')
    return violations
