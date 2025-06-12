function [cls, cls_category, available_vid] = get_action_name(action_id)
switch action_id
    case 11
        cls = 'Swing-Bench';
        cls_category = 1;
        available_vid = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010'};
    case 12
        cls = 'lifting';
        cls_category = 1;
        available_vid = {'001', '002', '003', '004', '005', '006'};
    case 13
        cls = 'Diving-Side';
        cls_category = 1; 
        available_vid = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010'};
    case 14
        cls = 'Golf-Swing-Side';
        cls_category = 1;
        available_vid = {'001', '002', '003', '004', '005'};
    case 15
        cls = 'Kicking-Front';
        cls_category = 1; 
        available_vid = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010'};
    case 16
        cls = 'run-side';
        cls_category = 1;
        available_vid = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013'};
    case 17
        cls = 'skateboarding-front';
        cls_category = 1;
        available_vid = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012'};
    case 18
        cls = 'walk-front';
        cls_category = 1;
        available_vid = {'001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022'};
    otherwise
        error('Unknown action_id: %d', action_id);
end