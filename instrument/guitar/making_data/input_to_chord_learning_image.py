import pandas as pd


# 현재 이 파일은 쓰이지 않는 중이지만 차후에 필요할 예정이여서 기록
def transform_chord_list_to_chord_audio_input_data(input_data, fps):
    get_data = pd.read_csv(input_data, na_values='NA', encoding='utf8')
    first_count = get_data['start'][0]
    get_data['time'] =get_data['end']-  get_data['start']
    transform_data = get_data.drop(columns=['start', 'end'])
    temp_data = pd.DataFrame({'chord': ['none'], 'time': [first_count]})
    refined_data = pd.concat([temp_data, transform_data], axis=0, ignore_index=True)
    refined_data['frame'] = fps * refined_data['time']
    reformed_data = refined_data.drop(columns=['time'])

    half_round = 0
    temp = 0
    ban_round_list = list()

    for row_count in range(len(reformed_data)):
        ban_new = round(reformed_data['frame'][row_count]) + temp

        diff = ban_new - reformed_data['frame'][row_count]
        # 올림하면 + , 내림하면 -

        ban_round_list += [ban_new]

        half_round = half_round + diff

        if half_round >= 1:
            temp = -1
            half_round = half_round - 1
        elif half_round <= -1:
            temp = 1
            half_round = half_round + 1
        else:
            temp = 0

    reformed_data['half_round'] = ban_round_list
    input_to_audio = reformed_data.drop(columns=['frame'])

    chord_list = []

    for num in range(len(input_to_audio)):
        row_count = 1
        while row_count <= input_to_audio['half_round'][num]:
            chord_list.append(input_to_audio['chord'][num])
            row_count += 1

    return input_to_audio