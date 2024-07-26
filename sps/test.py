# light.py - decides whether a Cayley table represents an associative operation
#
# Copyright 2015 Jeffrey Finkelstein.
#
# This file is part of Light.
#
# Light is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Light is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Light.  If not, see <http://www.gnu.org/licenses/>.
"""Provides a function for deciding whether an operation is associative, given
its Cayley table.

"""

__all__ = ['is_associative']


def cayley_table(elements, op):
    """Creates the Cayley table for the given binary operation.

    `elements` must be a set of objects. `op` must be a function that takes two
    inputs and produces a single output, each of which must be an object from
    `elements`.

    """
    # TODO This can be parallelized by a `parallel_dict()` constructor,
    # assuming the function `func` doesn't do anything strange.
    return {x: {y: op(x, y) for y in elements} for x in elements}


def is_associative(table):
    """Returns ``True`` if and only if the operation representation by the
    given Cayley table is associative.

    `table` must be the Cayley table of a `magma`_, represented as a dictionary
    of dictionaries. The table must be square (that is, it must satisfy
    ``all(set(table) == set(row) for row in table)``).

    This function implements the algorithm implicit in `Light's associativity
    test`_.

    .. _magma: https://en.wikipedia.org/wiki/Magma_%28algebra%29
    .. _Light's associativity test: https://en.wikipedia.org/wiki/Light's_associativity_test

    """
    S = set(table)
    left = lambda a: (lambda x, y: table[x][table[a][y]])
    right = lambda a: (lambda x, y: table[table[x][a]][y])
    # TODO This can be parallelized via a `parallel_all` function.
    return all(cayley_table(S, left(a)) == cayley_table(S, right(a))
               for a in S)

def main():
    # This is the left zero semigroup, LO_3.
    table = {'P': {'P': 'P', 'A': 'P', 'I': 'I'},
             'A': {'P': 'P', 'A': 'A', 'I': 'I'},
             'I': {'P': 'P', 'A': 'I', 'I': 'I'}}
    print(is_associative(table))

if __name__ == '__main__':
    main()