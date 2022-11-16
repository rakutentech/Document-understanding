#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/11/16 12:23
# @Author  : strawsyz
# @File    : CellBasedLayout.py
# @desc:


def optimize_coordinates(coordinates, threshold):
    """
    Give the same coordinates to coordinates that are less than threshold away
    :param coordinates: a sorted sequence of coordinates
    :param threshold:
    :return:
    """
    new_coordinates = [coordinates[0]]
    for idx, coo in enumerate(coordinates[:-1]):
        if coordinates[idx + 1] - coo > threshold:
            new_coordinates.append(coordinates[idx + 1])
    return new_coordinates


def create_coo_2_cell_map(coordinates, max_coo):
    """
    create a list to store cell information
    using coordinates as the index and using cell index as the value
    :param coordinates: a sorted sequence of coordinates
    :param max_coo: maximum coordinate value
    :return:
    """
    cell_layout = [0 for i in range(max_coo)]
    if len(coordinates) == 1:
        cell_layout[coordinates[0]:] = [1 for i in range(max_coo - coordinates[0])]
        return cell_layout

    pre_value = 0
    for idx, value in enumerate(coordinates[1:]):
        value = int(value)
        for i in range(pre_value, value):
            cell_layout[i] = idx
        pre_value = value

    cell_layout[value:] = [idx + 1 for i in range(max_coo - value)]

    return cell_layout


def get_cell_coo(cell_coos, cell_index, max_coo):
    """
    get the coordinate range of the cell
    :param cell_coos:
    :param cell_index: the index of the cell
    :param max_coo: the maximum value of the coordinates
    :return:
    """
    if len(cell_coos) == 1:
        return [cell_coos[0], max_coo]

    if cell_index > len(cell_coos) - 2:
        start = cell_coos[cell_index]
        end = max_coo
    else:
        start = cell_coos[cell_index]
        end = cell_coos[cell_index + 1]
    return [start, end]


def create_cell_layout_4_layoutlm(bboxes, max_x, max_y, scale=0.01):
    """
    create a cell-based layout according to all bboxes
    :param bboxes: bounding boxes
    :param max_x: the maximum value of the x-coordinate
    :param max_y: the maximum value of the y-coordinate
    :param scale: to control numbder of cells in a document.
                use scale*max_x or scale*max_y as the threshold
    :return: return a bbox [x1, y1, x3, y3] and [row index, column index] for each cell
    """

    # Use key point the calculate number of rows and columns
    row_coos, col_coos = set(), set()
    bbox_of_cells = []
    index_of_cells = []
    for bbox in bboxes:
        key_point = bbox[:2]
        row_coos.add(key_point[1])
        col_coos.add(key_point[0])

    # sort x coordinates and y coordinates
    row_coos = list(row_coos)
    row_coos.sort()
    col_coos = list(col_coos)
    col_coos.sort()

    # optimize coordinates
    row_coos = optimize_coordinates(row_coos, max_y * scale)
    col_coos = optimize_coordinates(col_coos, max_x * scale)

    row_coo_2_cell_map = create_coo_2_cell_map(row_coos, max_y)
    col_coo_2_cell_map = create_coo_2_cell_map(col_coos, max_x)

    # give every bbox a cell number
    for bbox in bboxes:
        key_point = bbox[:2]
        row_cell = row_coo_2_cell_map[int(key_point[1])]
        col_cell = col_coo_2_cell_map[int(key_point[0])]

        y1, y3 = get_cell_coo(row_coos, row_cell, max_y)
        x1, x3 = get_cell_coo(col_coos, col_cell, max_x)
        bbox_of_cells.append([x1, y1, x3, y3])
        index_of_cells.append([row_cell, col_cell])

    return bbox_of_cells, index_of_cells

