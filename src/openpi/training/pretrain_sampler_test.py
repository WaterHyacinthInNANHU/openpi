from __future__ import annotations

from openpi.training import pretrain_sampler as ps


def test_single_episode_range_maps_to_global_rows():
    # One sub-dataset (task 800) placed at global offset 0, one episode 0 starting at
    # local row 0 with length 10. A single kept range [2, 5) selects frames 2,3,4.
    sub = ps.SubDataset(task_id=800, global_base=0, episodes={0: (0, 10)})
    ranges = {"task_800--0": [[2, 5]]}
    assert ps.plan_pretrain_rows([sub], ranges) == [2, 3, 4]


def test_range_end_is_clamped_to_episode_length():
    # A range that runs past the episode end must not emit rows beyond the last frame.
    sub = ps.SubDataset(task_id=800, global_base=0, episodes={0: (0, 4)})
    ranges = {"task_800--0": [[2, 99]]}
    assert ps.plan_pretrain_rows([sub], ranges) == [2, 3]


def test_negative_range_start_is_clamped_to_zero():
    sub = ps.SubDataset(task_id=800, global_base=0, episodes={0: (0, 5)})
    ranges = {"task_800--0": [[-3, 2]]}
    assert ps.plan_pretrain_rows([sub], ranges) == [0, 1]


def test_episode_without_a_ranges_key_contributes_nothing():
    sub = ps.SubDataset(task_id=800, global_base=0, episodes={0: (0, 5), 1: (5, 5)})
    ranges = {"task_800--0": [[0, 2]]}  # nothing for episode 1
    assert ps.plan_pretrain_rows([sub], ranges) == [0, 1]


def test_local_start_and_multiple_ranges_within_one_episode():
    # Episode 3 does not start at row 0; local_start offsets every frame. Two disjoint
    # ranges both contribute, in range order.
    sub = ps.SubDataset(task_id=801, global_base=0, episodes={3: (100, 20)})
    ranges = {"task_801--3": [[0, 2], [10, 12]]}
    assert ps.plan_pretrain_rows([sub], ranges) == [100, 101, 110, 111]


def test_global_base_offsets_second_subdataset():
    # Two tasks concatenated: task 800 occupies rows [0, 10); task 900 starts at
    # global_base=10. Frame f of task 900's episode 0 maps to 10 + f.
    a = ps.SubDataset(task_id=800, global_base=0, episodes={0: (0, 10)})
    b = ps.SubDataset(task_id=900, global_base=10, episodes={0: (0, 8)})
    ranges = {"task_800--0": [[1, 3]], "task_900--0": [[0, 2]]}
    assert ps.plan_pretrain_rows([a, b], ranges) == [1, 2, 10, 11]


def test_episode_key_format_matches_pretrain_ranges():
    # The ranges file is emitted by axis_data.pretrain_ranges with exactly this
    # key format; drift here silently drops every row. Pin it.
    assert ps.episode_key(1644, 7) == "task_1644--7"


def test_build_subdatasets_assigns_cumulative_global_bases():
    # torch ConcatDataset concatenates by len(): sub-dataset k starts at the summed length
    # of all earlier sub-datasets. Given per-task (task_id, episodes, length) in order, the
    # builder fills global_base cumulatively and reports the total row count.
    entries = [
        (800, {0: (0, 6), 1: (6, 4)}, 10),  # length 10
        (900, {0: (0, 5)}, 5),  # starts at 10
    ]
    subs, total = ps.build_subdatasets(entries)
    assert [s.task_id for s in subs] == [800, 900]
    assert [s.global_base for s in subs] == [0, 10]
    assert subs[0].episodes == {0: (0, 6), 1: (6, 4)}
    assert total == 15


def test_build_subdatasets_then_plan_rows_end_to_end():
    # The two pure pieces compose: build bases, then plan rows across tasks.
    entries = [(800, {0: (0, 10)}, 10), (900, {0: (0, 8)}, 8)]
    subs, _ = ps.build_subdatasets(entries)
    ranges = {"task_800--0": [[1, 3]], "task_900--0": [[0, 2]]}
    assert ps.plan_pretrain_rows(subs, ranges) == [1, 2, 10, 11]
