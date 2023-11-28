%% Convert script for Regression - Original Data
%% convert matrix of vectors to matrix - EP conn - train - 41, test - 10 patients
% No Augmentation

clear all
close all
clc

%% Load data
dir = 'C:\Users\soumy\Dropbox (Personal)\Wayne_State_PhD\Lab_work\Connectome_regression_data\';
density = '04';
filename_expr = strcat('X_expressive_q_',density,'.mat');
filepath_expr = strcat(dir,filename_expr);
load(filepath_expr);

filename_recep = strcat('X_receptive_q_',density,'.mat');
filepath_recep = strcat(dir,filename_recep);
load(filepath_recep);

%% Construct Training Data

%% expressive score data

x_train_expressive_mat = zeros(116,116,41);  
y_train_expressive_mat = zeros(41,1);

for i=1:41
    x_train_expressive_mat(:,:,i) = vec_to_mat(X_expressive(i,1:6728));   
    y_train_expressive_mat(i,1) = X_expressive(i,6729);    
end

%% receptive score data

x_train_receptive_mat = zeros(116,116,41);  
y_train_receptive_mat = zeros(41,1);

for i=1:41
    x_train_receptive_mat(:,:,i) = vec_to_mat(X_receptive(i,1:6728));   
    y_train_receptive_mat(i,1) = X_receptive(i,6729);    
end

%% Construct Test Data

%% expressive score data

x_test_expressive_mat = zeros(116,116,10);  
y_test_expressive_mat = zeros(10,1);

for j = 42:51
    x_test_expressive_mat(:,:,j-41) = vec_to_mat(X_expressive(j,1:6728)); 
    y_test_expressive_mat(j-41,1) = X_expressive(j,6729);
end

%% receptive score data

x_test_receptive_mat = zeros(116,116,10);  
y_test_receptive_mat = zeros(10,1);

for j = 42:51
    x_test_receptive_mat(:,:,j-41) = vec_to_mat(X_receptive(j,1:6728)); 
    y_test_receptive_mat(j-41,1) = X_receptive(j,6729);
end

%% Save Data

%% expressive data

save_dir = 'C:\Users\soumy\Dropbox (Personal)\Wayne_State_PhD\Lab_work\Connectome_regression_data\processed_data\expressive_data\original_data\';

% save the augmented train and test expressive data
save(strcat(save_dir,'x_train_expressive_density_',density,'.mat'),'x_train_expressive_mat','-v7.3');
save(strcat(save_dir,'y_train_expressive_density_',density,'.mat'),'y_train_expressive_mat');

save(strcat(save_dir,'x_test_expressive_density_',density,'.mat'),'x_test_expressive_mat');
save(strcat(save_dir,'y_test_expressive_density_',density,'.mat'),'y_test_expressive_mat');


% fighandle = figure;
% set(fighandle,'Position',[200,200,116*5,116*5],'Resize','off');    
% imagesc(x_train_expressive_mat(:,:,1))
% title('Expressive Connectome with density 8')
% 
% fighandle = figure;
% set(fighandle,'Position',[200,200,116*5,116*5],'Resize','off');
% imagesc(x_test_expressive_mat(:,:,20))

%% receptive data

save_dir = 'C:\Users\soumy\Dropbox (Personal)\Wayne_State_PhD\Lab_work\Connectome_regression_data\processed_data\receptive_data\original_data\';

% save the augmented train and test receptive data
save(strcat(save_dir,'x_train_receptive_density_',density,'.mat'),'x_train_receptive_mat','-v7.3');
save(strcat(save_dir,'y_train_receptive_density_',density,'.mat'),'y_train_receptive_mat');

save(strcat(save_dir,'x_test_receptive_density_',density,'.mat'),'x_test_receptive_mat');
save(strcat(save_dir,'y_test_receptive_density_',density,'.mat'),'y_test_receptive_mat');

