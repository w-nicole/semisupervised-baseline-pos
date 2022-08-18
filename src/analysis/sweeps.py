
import plotly.express as px
import os
import pandas as pd
import numpy as np

def get_heatmap(df, row_key, column_key, value_key, analysis_path):
    
    to_sorted_tick = lambda params : list(map(lambda s: str(s), sorted(list(set(params)))))
    df = df.sort_values(by=[row_key, column_key], ascending = True)

    row_values = to_sorted_tick(df[row_key])
    column_values = to_sorted_tick(df[column_key])
    result_values = df[value_key]
    
    number_of_rows = len(row_values)
    number_of_columns = len(column_values)
    
    raw_scores = np.array(result_values).reshape((number_of_rows, number_of_columns))
    rounded_scores = np.round_(raw_scores, decimals = 1)
    
    for modifier, scores in zip(['', 'rounded_'], [raw_scores, rounded_scores]):
        fig = px.imshow(
            scores, text_auto = True,
            labels = {'y' : row_key, 'x' : column_key, 'color' : value_key},
            y = row_values, x = column_values,
            color_continuous_scale = 'rainbow'
        )
        if not os.path.exists(analysis_path): os.makedirs(analysis_path)
        fig.write_html(os.path.join(analysis_path, f'{modifier}{value_key}.html'))
    return raw_scores
    
def find_hparam(hparam_name, hparam_list, cast_as):
    hparam_prefix = f'--{hparam_name}='
    matches = list(filter(lambda s : hparam_prefix in s, hparam_list))
    assert len(matches) == 1, f'{hparam_name}, {hparam_list}, {matches}'
    return cast_as(matches[0].replace(hparam_prefix, ""))
    