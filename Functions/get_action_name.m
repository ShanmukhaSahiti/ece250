function [cls, cls_category, available_vid] = get_action_name(action_id)
if action_id == 11
    cls = 'Swing-Bench';
    cls_category = 1;
    available_vid = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010'};
elseif action_id == 12
    cls = 'lifting';
    cls_category = 1;
    available_vid = {'v1', '2502-2', '001', '002', '003', '004', '005', '006'};
elseif action_id == 13
    cls = 'Diving-Side';
    cls_category = 1; 
    available_vid = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010'};
elseif action_id == 14
    cls = 'Golf-Swing-Side';
    cls_category = 1;
    available_vid = {'001', '002', '003', '004', '005'};
elseif action_id == 15
    cls = 'Kicking-Front';
    cls_category = 1;
    available_vid = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010'};
else
    error('Unknown action_id: %d', action_id);
end