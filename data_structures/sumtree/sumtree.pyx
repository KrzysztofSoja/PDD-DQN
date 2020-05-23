import numpy as np
import random as rand
from cpython.mem cimport PyMem_Malloc, PyMem_Free


cdef set_zero_in_c_table(int* table, int size):
    """
    Set zero on all position in C table.
    :param table: pointer to C-style int table
    :param size: size of table
    """
    cdef int idx = 0
    while idx < size:
        table[idx] = 0
        idx += 1

cdef class SumTree:
    """
    SumTree is data structure, that store items with probability. This structures allows sampling items
    and adding new elements to tree after initialize structures. Items is storing only in leafs. Measure of probability
    must be natural number. Size of probability space is storing in the root. SumTree is circular buffer.
    When number of elements in tree is bigger than capacity, adding new items overload the eldest item.

    Time:
        -> initialize empty sumtree O(n)
        -> sampling O(log(n))
        -> add new elements O(log(n))
    """

    cdef int elements_in_tree
    cdef int number_of_nodes
    cdef int capacity
    cdef int first_leaf_idx
    cdef int* tree
    cdef items

    def __cinit__(self, capacity: int):
        self.capacity = capacity
        self.number_of_nodes = 2*capacity - 1
        self.first_leaf_idx = self.number_of_nodes - self.capacity
        self.elements_in_tree = 0

        self.tree = <int*> PyMem_Malloc(self.number_of_nodes * sizeof(int))
        if not self.tree:
            raise MemoryError()
        set_zero_in_c_table(self.tree, self.number_of_nodes)

        self.items =  np.zeros((capacity, ), dtype=object)

    def __dealloc__(self):
        PyMem_Free(self.tree)

    cdef _back_propagate(self, int number, int actual_node_idx):
        """
        Modify intervals storing in tree nodes. Propagation occurs from leaf to root.
        :param number: This number is adding to node's intervals.
        :param actual_node_idx: Leaf idx.
        :return: 
        """

        self.tree[actual_node_idx] += number

        while actual_node_idx > 0:
            if actual_node_idx % 2 == 0:
                actual_node_idx =  actual_node_idx/2  - 1
            else:
                actual_node_idx = actual_node_idx/2
            self.tree[actual_node_idx] += number


    def add(self, key: int, item: object):
        """
        Add new item to tree.
        :param key: probability range
        :param item: elements that you want add to tree
        :return:
        """

        if self.elements_in_tree == self.capacity:
            self.elements_in_tree = 0
        self.items[self.elements_in_tree] = item

        cdef int idx = self.number_of_nodes - self.elements_in_tree - 1
        cdef int update = key - self.tree[idx]

        self._back_propagate(update, idx)
        self.elements_in_tree += 1

    cdef _go_to_leaf(self, int key):
        """
        Move to leaf with probability range suitable for key.
        :param key: Must be number from interval suitable for the given item.
        :return: Index of leaf with items.
        """
        cdef int parent = 0
        cdef int left_child = 2*parent + 1
        cdef int right_child = 2*parent + 2

        while parent < self.first_leaf_idx:
            if key <= self.tree[left_child]:
                parent = left_child
            else:
                parent = right_child
                key -= self.tree[left_child]

            left_child = 2*parent + 1
            right_child = 2*parent + 2

        return parent

    def get_size_of_probability_space(self) -> int:
        return self.tree[0]

    def get(self, key: int) -> object:
        """
        :param key: Must be number from interval suitable for the given item. Interval maybe be changed during work.
        :return: Item storing in sumtree.
        """
        cdef int leaf_idx = self._go_to_leaf(key)
        return self.items[leaf_idx - self.first_leaf_idx]

    def sample(self) -> object:
        """
        :return: Random element form tree.
        """
        cdef int key = rand.randint(0, self.tree[0] - 1)
        return self.get(key)

    def get_item_table(self) -> np.array:
        """
        Using only in tests.
        :return: Table storing items.
        """
        return self.items

    def get_tree_table(self) -> np.array:
        """
        Using only in tests.
        :return: Table storing nodes values.
        """
        tree_table =np.empty((self.number_of_nodes,), dtype=int)
        for idx in range(self.number_of_nodes):
            tree_table[idx] = self.tree[idx]
        return tree_table