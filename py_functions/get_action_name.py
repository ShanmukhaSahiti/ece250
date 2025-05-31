import numpy as np

def get_action_name(action_id):
    if action_id == 5:
        cls_name = 'lateral lunge'
        cls_category = 2
        available_vid = ['2', '5']
    elif action_id == 9:
        cls_name = 'sit up'
        cls_category = 8
        available_vid = ['3-4', '6']
    elif action_id == 10:
        cls_name = 'stiff-leg deadlift'
        cls_category = 1
        available_vid = ['4', '6']
    else:
        raise ValueError('Activity not found. Please choose one of the sample provided activities (5, 9, or 10).')
    return cls_name, cls_category, available_vid 