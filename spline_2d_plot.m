function [y,x1,x2] = spline_2d_plot(param,cpts,s)

%% This is adapted from Uri Eden's spline lab.

% y = the values of the variable over time
% cpts = control points
% s = tension parameters

% Here I will assume that points are equally spaced (can do Catmull-Rom
% splines)

% S matrix
S = [-s 2-s s-2 s;2*s s-3 3-2*s -s;-s 0 s 0;0 1 0 0];

param = reshape(param,numel(param),1);

% Lay down extra control points
%cpts_x = cpts{1};
%cpts_y = cpts{2};
%bin_x = cpts_x(2)-cpts_x(1);
%bin_y = cpts_y(2)-cpts_y(1);
%cpts_x_all = [cpts_x(1)-bin_x cpts_x cpts_x(end)+bin_x];
%cpts_y_all = [cpts_y(1)-bin_y cpts_y cpts_y(end)+bin_y];
cpts_x_all = cpts{1};
cpts_y_all = cpts{2};
num_c_pts_x = length(cpts_x_all);  %number of control points in total
num_c_pts_y = length(cpts_y_all); 

bin_x = cpts_x_all(2)-cpts_x_all(1);
bin_y = cpts_y_all(2)-cpts_y_all(1);

x1 = 0:min(cpts_x_all(end)-bin_x,75);
x2 = 0:min(cpts_y_all(end)-bin_y,75);
y = nan(length(x2),length(x1));

%for each timepoint, calculate the corresponding row of the glm input matrix
for i=1:length(x1)
    
    % for 1st dimension
    % find the nearest, and next, control point
    nearest_c_pt_index_1 = max(find(cpts_x_all < x1(i)));
    nearest_c_pt_time_1 = cpts_x_all(nearest_c_pt_index_1);
    next_c_pt_time_1 = cpts_x_all(nearest_c_pt_index_1+1);
    
    % compute the alpha (u here)
    u_1 = (x1(i)-nearest_c_pt_time_1)/(next_c_pt_time_1-nearest_c_pt_time_1);
    p_1=[u_1^3 u_1^2 u_1 1]*S;
    
    % fill in the X matrix, with the right # of zeros on either side
    X1 = [zeros(1,nearest_c_pt_index_1-2) p_1 zeros(1,num_c_pts_x-4-(nearest_c_pt_index_1-2))];
    
    for j = 1:length(x2)
        
        
        % for 2nd dimension
        % find the nearest, and next, control point
        nearest_c_pt_index_2 = max(find(cpts_y_all < x2(j)));
        nearest_c_pt_time_2 = cpts_y_all(nearest_c_pt_index_2);
        next_c_pt_time_2 = cpts_y_all(nearest_c_pt_index_2+1);
        
        % compute the alpha (u here)
        u_2 = (x2(j)-nearest_c_pt_time_2)/(next_c_pt_time_2-nearest_c_pt_time_2);
        p_2=[u_2^3 u_2^2 u_2 1]*S;
        
        % fill in the X matrix, with the right # of zeros on either side
        X2 = [zeros(1,nearest_c_pt_index_2-2) p_2 zeros(1,num_c_pts_y-4-(nearest_c_pt_index_2-2))];
        
        % take the outer product
        X12_op = X2'*X1; X12_op = flipud(X12_op);
        
        X = X12_op(:);
        
        y(length(x2) - j + 1,i) = X'*param;

 
    end
    
    
end
