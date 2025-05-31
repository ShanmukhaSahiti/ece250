function [localX, localY, localZ] = get_local_coordinate_system(verts, torso_vts_idx,front_torso_idx,back_torso_idx, left_toes_idx, show)

torso_vts = verts(torso_vts_idx,:);
torso_centroid = mean(torso_vts);
[rot_matrix,DD]= eig((torso_vts-torso_centroid)' * (torso_vts-torso_centroid));
[~,axis_strength] = sort(diag(DD),'descend');


localX = rot_matrix(:,axis_strength(3)); %axis pointing to front of person, should align with +x axis
if (verts(front_torso_idx,:) - verts(back_torso_idx,:))*localX < 0
    localX= -localX;
end

localZ = rot_matrix(:,axis_strength(1));%axis pointing to above the person, should align with -x axis
if (verts(front_torso_idx,:) - verts(left_toes_idx(1),:))*localZ < 0
    localZ = -localZ;
end

localY = cross(localZ, localX);

if show
    scatter3(verts(:,1), verts(:,2), verts(:,3),'.')
    hold on,
    quiver3(torso_centroid(1),torso_centroid(2),torso_centroid(3), localX(1), localX(2), localX(3), 'r')
    hold on,
    quiver3(torso_centroid(1),torso_centroid(2),torso_centroid(3), localY(1), localY(2), localY(3), 'g')
    hold on,
    quiver3(torso_centroid(1),torso_centroid(2),torso_centroid(3), localZ(1), localZ(2), localZ(3), 'k')
end
