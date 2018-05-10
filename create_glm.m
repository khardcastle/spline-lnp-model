function [bestModels,tuning_curves,final_pval, fig1] = create_glm(x,y,head_direction,speed,spiketrain,boxSize,dt)

%% ASSUMPTIONS:
% x and y are in real units (cm) that go from 0 to the boxsize (everything
% is re-scaled

% head direction is in units of radians (go from 0 to 2*pi)


%% do some initial processing 
% remove NaNs from head direction, eye in head direction

var_name = {'position','hd','speed'};
numVar = numel(var_name);


% make sure x and y start at 0
x = x - min(x);
y = y - min(y);

isnan_x = find(isnan(x));
isnan_y = find(isnan(y));
isnan_h = find(isnan(head_direction));
isnan_s = find(isnan(speed));
isnan_all = [isnan_x; isnan_y; isnan_h; isnan_s];

spiketrain(isnan_all) = [];
x(isnan_all) = [];
y(isnan_all) = [];
head_direction(isnan_all) = [];
speed(isnan_all) = [];

%% compute the input matrices for head direction and gaze

fig1 = figure(1);
s = 0.5; % spline parameter
%%%%%%%%% POSITION %%%%%%%%%
    fprintf('Making position matrix\n');
    
    % plot position coverage and position tuning curve
    
    bin_x = round(max(x)*(25/100));
    bin_y = round(max(y)*(25/100));
    [pos_tuning_curve,pos_occupancy] = compute_2d_tuning_curve(x,y,spiketrain,bin_x,0,max(x),bin_y,0,max(y));

    fig1 = subplot(4,numVar,1);
    imagesc(pos_occupancy.*dt)
    axis off
    title('position occupancy')
    caxis([0 50])
    colorbar
    axis tight
    
    fig1 = subplot(4,numVar,1+numVar);
    imagesc(pos_tuning_curve./dt)
    axis off
    title('position tuning curve')
    colorbar
    ylabel('spikes/s')
    axis tight
    
    bin_ratio = 10/100;
    bin_x = round(max(x)*bin_ratio);
    bin_y = round(max(y)*bin_ratio);
    x_vec = linspace(0,max(x),bin_x); x_vec(1) = -0.01;
    y_vec = linspace(0,boxSize,bin_y); y_vec(1) = -0.01;
    [posgrid,pos_x_pts,pos_y_pts] = spline_2d(x,y,x_vec,y_vec,s);
    A{1} = posgrid;
    ctl_pts_all{1} = {pos_x_pts,pos_y_pts};
%%%%%%%%% HEAD DIRECTION %%%%%%%%%
    
    % plot coverage and tuning curve
    [hd_tuning_curve,hd_occupancy] = compute_1d_tuning_curve(head_direction,spiketrain,18,0,2*pi);

    fig1 = subplot(4,numVar,2);
    plot(linspace(0,2*pi,18),hd_occupancy.*dt,'k','linewidth',2)
    box off
    title('hd occupancy')
    ylabel('seconds')
    axis tight
    
    fig1 = subplot(4,numVar,2+numVar);
    plot(linspace(0,2*pi,18),hd_tuning_curve./dt,'k','linewidth',2)
    box off
    title('hd tuning curve')
    ylabel('spikes/s')
    axis tight
    
    fprintf('Making head direction matrix\n');
    bin_h = 12;
    hd_vec = linspace(0,2*pi,bin_h+1); hd_vec = hd_vec(1:end-1);
    [hdgrid] = spline_1d_circ(head_direction,hd_vec,s);
    A{2} = hdgrid;
    ctl_pts_all{2} = hd_vec;

%%%%%%%%% SPEED %%%%%%%%%
    
    % plot coverage and tuning curve
    sorted_speed = sort(speed);
    max_speed = ceil(sorted_speed(round(numel(speed)*0.99))/10)*10;
    [speed_tuning_curve,speed_occupancy] = compute_1d_tuning_curve(speed,spiketrain,10,0,max_speed);

    fig1 = subplot(4,numVar,3);
    plot(linspace(0,max_speed,10),speed_occupancy.*dt,'k','linewidth',2)
    box off
    title('speed occupancy')
    axis tight
    ylabel('seconds')
    
    fig1 = subplot(4,numVar,3+numVar);
    plot(linspace(0,max_speed,10),speed_tuning_curve./dt,'k','linewidth',2)
    box off
    title('speed tuning curve')
    axis tight
    ylabel('spikes/s')
    
    
    fprintf('Making speed matrix\n');
    spdVec = [0:5:max_speed]; spdVec(end) = max_speed; spdVec(1) = -0.1;
    speed(speed > max_speed) = max_speed; %send everything over max to max
    [speedgrid,~] = spline_1d(speed,spdVec,s);
    A{3} = speedgrid;
    ctl_pts_all{3} = spdVec;


%% fit the model

%%%%%%% COMPUTE TEST AND TRAIN INDICES %%%%%
numFolds = 10;
T = numel(spiketrain); 
numPts = 3*round(1/dt); % 3 seconds. i've tried #'s from 1-10 seconds.. not sure what is best
[train_ind,test_ind] = compute_test_train_ind(numFolds,numPts,T);

%%%%%%%% FORWARD SEARCH PROCEDURE %%%%%%%%%
[allModelTestFits, allModelTrainFits, bestModels, bestModelFits, parameters, pvals, final_pval] = forward_search_kfold(A,spiketrain,train_ind,test_ind);

% plot the model tuning curves (this is an approx)
plotfig = 1;
final_param = parameters{end};
[tuning_curves,fig1] = plot_tuning(A,bestModels,final_param,ctl_pts_all,s,plotfig,dt,fig1,var_name);

firstModelFit = allModelTestFits{1};
fig1 = subplot(4,numVar,numVar*3+1:numVar*3+numVar);
errorbar(1:numVar,mean(firstModelFit),std(firstModelFit)/sqrt(10),'.k','linewidth',2)
hold on
plot([1 numVar],[0 0],'--b','linewidth',1);
hold off
box off
set(gca,'xtick',[1 2 3])
set(gca,'xticklabel',{'position','hd','speed'})
ylabel('bits/spike')
axis([0.5 3.5 -inf inf])
legend('first model fit','baseline')

return