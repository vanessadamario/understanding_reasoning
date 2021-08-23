dict_separated_tasks = {0: {'count': 4,
                            'equal_integer': 6,
                            'exist': 10,
                            'greater_than': 26,
                            'less_than': 28,
                            'unique': 43},
                        1: {'equal_color': 5,
                            'filter_color[blue]': 11,
                            'filter_color[brown]': 12,
                            'filter_color[cyan]': 13,
                            'filter_color[gray]': 14,
                            'filter_color[green]': 15,
                            'filter_color[purple]': 16,
                            'filter_color[red]': 17,
                            'filter_color[yellow]': 18,
                            'query_color': 29,
                            'same_color': 3},
                        2: {'equal_material': 7,
                            'filter_material[metal]': 19,
                            'filter_material[rubber]': 20,
                            'query_material': 30,
                            'same_material': 38},
                        3: {'equal_shape': 8,
                            'filter_shape[cube]': 21,
                            'filter_shape[cylinder]': 22,
                            'filter_shape[sphere]': 23,
                            'query_shape': 31,
                            'same_shape': 39},
                        4: {'equal_size': 9,
                            'filter_size[large]': 24,
                            'filter_size[small]': 25,
                            'query_size': 32,
                            'same_size': 40},
                        5: {'relate[behind]': 33,
                            'relate[front]': 34,
                            'relate[left]': 35,
                            'relate[right]': 36},
                        6: {'intersect': 27,
                            'union': 42},
                        7: {'scene': 41}
                        }


def invert_task_div(dict_separated_tasks):
    dct_program_token = {}
    dct_program_idx = {}

    for k_group, v_group in dict_separated_tasks.items():
        for k_prog, v_prog in v_group.items():
            dct_program_token[k_prog] = k_group
            dct_program_idx[v_prog] = k_group

    map_id_subtasks = [dct_program_idx[jj] if jj in dct_program_idx.keys() else -1
                       for jj in range(44)]

    return map_id_subtasks


def invert_task_questions(dict_separated_tasks, vocab):
    dct_program_token = {}
    dct_program_idx = {}

    for k_group, v_group in dict_separated_tasks.items():
        for k_prog, v_prog in v_group.items():
            dct_program_token[k_prog] = k_group
            dct_program_idx[v_prog] = k_group

    map_id_subtasks = [dct_program_idx[jj] if jj in dct_program_idx.keys() else -1
                       for jj in range(44)]

    return map_id_subtasks
