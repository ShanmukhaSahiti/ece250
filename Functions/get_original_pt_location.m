function y = get_original_pt_location(x, person_box, cams, im_shape)

%this function takes a point x in the bounding box and returns its location
% in the original video frame
margin_expand = 40;
x_min = max([person_box(2)-margin_expand, 0]);
y_min = max([person_box(1)-margin_expand, 0]);
x_max = min([person_box(4)+margin_expand, im_shape(2)-1]);
y_max = min([person_box(3)+margin_expand, im_shape(1)-1]);



largest_dim = max(y_max-y_min,x_max-x_min);
smallest_dim = min(y_max-y_min,x_max-x_min);
x(:,1) = x(:,1) + cams(2);
x(:,3) = -x(:,3) + cams(3);
x = x*largest_dim/2;

if y_max - y_min >= x_max - x_min
    x(:,3) = x(:,3) + y_min + largest_dim/2;
    x(:,1) = x(:,1) + x_min + smallest_dim/2;
else
    x(:,1) = x(:,1) + x_min + largest_dim/2;
    x(:,3) = x(:,3) + y_min + smallest_dim/2;
    
end

% x(:,2) = - x(:,2);

% x pos (col), y pos (row)
y = [x(:,1), x(:,3)];

