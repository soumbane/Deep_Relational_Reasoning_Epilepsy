%% Convert Script for binary classification - Seizure outcome (1-SZ-free, 0-has SZ) - Fold 1
%% convert matrix of vectors to matrix - EP conn - train - 20040, test - 5010 patients
% 40 training patients - 40*500 + 40 = 20040 training connectomes
% 10 validation/testing patients - 10*500 + 10 = 5010 total testing connectomes
% after augmentation

clear all
close all
clc

%% Load data
dir = 'C:\Users\soumy\Dropbox (Personal)\Wayne_State_PhD\Lab_work\Connectome_classification_data\';

density = '04';
filename_expr = strcat('X_sz_q_',density,'.mat');
filepath_expr = strcat(dir,filename_expr);
load(filepath_expr);


%% Construct Training Data

x_train_mat = zeros(116,116,20040); % 20040 patients with 500 aug data for each patient (500*40 + 40) 

y_train_mat = zeros(20040,1); % train labels (1-SZ-free, 0-has SZ)

% Form the augmented training set - 40*501 = 20040 training samples
% original connectomes

for i=11:50
    x_train_mat(:,:,i-10) = vec_to_mat(X_sz(i,1:6728));   
    y_train_mat(i-10,1) = X_sz(i,6729);    
end

% augmented connectomes
start = ((50+500*10)+1); % start of augmented train set

for i=start:25050
    x_train_mat(:,:,(i-(start-40))+1) = vec_to_mat(X_sz(i,1:6728));   
    y_train_mat((i-(start-40))+1,1) = X_sz(i,6729);    
end

%% Construct Test Data

x_test_mat = zeros(116,116,5010); % 10*500 + 10 = 5010 patients 

y_test_mat = zeros(5010,1); % test labels

% original connectomes
for j = 1:10
    x_test_mat(:,:,j) = vec_to_mat(X_sz(j,1:6728)); 
    y_test_mat(j,1) = X_sz(j,6729);
end

% augmented connectomes
end_test = ((50+500*10)); % end of augmented test set

for j = 51:end_test
    x_test_mat(:,:,j-40) = vec_to_mat(X_sz(j,1:6728)); 
    y_test_mat(j-40,1) = X_sz(j,6729);
end

%% Save Data

save_dir = 'C:\Users\soumy\Dropbox (Personal)\Wayne_State_PhD\Lab_work\Connectome_classification_data\new_classification_data\fold1\';

% save the augmented train and test expressive data
save(strcat(save_dir,'x_train_density_',density,'.mat'),'x_train_mat','-v7.3');
save(strcat(save_dir,'y_train_density_',density,'.mat'),'y_train_mat');

save(strcat(save_dir,'x_test_density_',density,'.mat'),'x_test_mat');
save(strcat(save_dir,'y_test_density_',density,'.mat'),'y_test_mat');


