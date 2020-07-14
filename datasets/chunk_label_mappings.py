movie_label_to_chunk_dict = {
    0: {"rat": [{"start": 100, "end": 3500}],
        "floor": [{"start": 3500, "end": 4300}, {"start": 22300, "end": 23700}],
        "duck": [{"start": 4300, "end": 7500}, {"start": 8100, "end": 9100}],
        "elephant_toy": [{"start": 9100, "end": 15300}],
        "apple": [{"start": 16700, "end": 22300}],
        "pear_apple": [{"start": 24100, "end": 26100}]},
    1: {"lion": [{"start": 100, "end": 3900}],
        "birds_over_lake": [{"start": 4900, "end": 7500}],
        "aerial_view": [{"start": 9300, "end": 20100}],
        "water_channel": [{"start": 21700, "end": 28000}]},
    2: {"elephant_lake": [{"start": 100, "end": 3500}],
        "elephants_lake": [{"start": 3540, "end": 8100}],
        "elephant_back": [{"start": 9900, "end": 14500}],
        "elephant_full": [{"start": 17500, "end": 28100}]}
}

movie_chunk_to_label_dict = {
    0: [{'start': 100, 'end': 3500, 'label': 'rat'},
        {'start': 3500, 'end': 4300, 'label': 'floor'},
        {'start': 22300, 'end': 23700, 'label': 'floor'},
        {'start': 4300, 'end': 7500, 'label': 'duck'},
        {'start': 8100, 'end': 9100, 'label': 'duck'},
        {'start': 9100, 'end': 15300, 'label': 'elephant_toy'},
        {'start': 16700, 'end': 22300, 'label': 'apple'},
        {'start': 24100, 'end': 26100, 'label': 'pear_apple'}],
    1: [{'start': 100, 'end': 3900, 'label': 'lion'},
        {'start': 4900, 'end': 7500, 'label': 'birds_over_lake'},
        {'start': 9300, 'end': 20100, 'label': 'aerial_view'},
        {'start': 21700, 'end': 28000, 'label': 'water_channel'}],
    2: [{'start': 100, 'end': 3500, 'label': 'elephant_lake'},
        {'start': 3540, 'end': 8100, 'label': 'elephants_lake'},
        {'start': 9900, 'end': 14500, 'label': 'elephant_back'},
        {'start': 17500, 'end': 28100, 'label': 'elephant_full'}]
}
