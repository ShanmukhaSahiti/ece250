function [cls, cls_category, available_vid] = get_action_name(action_id)
if action_id  == 5
    cls = 'lateral lunge';
    cls_category = 2;
    available_vid = {'2', '5'};
elseif action_id  == 9
    cls = 'sit up';
    cls_category = 8;
    available_vid = {'3-4','6'};
elseif action_id  == 10
    cls = 'stiff-leg deadlift';
    cls_category = 1;
    available_vid = {'4','6'};
elseif action_id == 11
    cls = 'swing bench';
    cls_category = 1;
    available_vid = {'1'};
else
    error('Unknown action_id: %d', action_id);
end