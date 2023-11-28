%% Convert Script for binary classification - Seizure outcome (1-SZ-free, 0-has SZ) - original data
%% convert matrix of vectors to matrix - EP conn - train - 40, test - 10 patients

clear all
close all
clc

%% Load data
dir = 'C:\Users\soumy\Dropbox (Personal)\Wayne_State_PhD\Lab_work\Connectome_for_Soumyanil_new_classification_data_July17_2019\';

density = '04';
filename_expr = strcat('X_sz_q_',density,'.mat');
filepath_expr = strcat(dir,filename_expr);
load(filepath_expr);


%% Construct Training Data

x_train_mat = zeros(116,116,40); % 40 patients 

y_train_mat = zeros(40,1); % train labels (1-SZ-free, 0-has SZ)

% original connectomes
for i=1:40
    x_train_mat(:,:,i) = vec_to_mat(X_sz(i,1:6728));   
    y_train_mat(i,1) = X_sz(i,6729);    
end

%% Construct Test Data

x_test_mat = zeros(116,116,10); % 10 patients 

y_test_mat = zeros(10,1); % test labels

% original connectomes
for j = 41:50
    x_test_mat(:,:,j-40) = vec_to_mat(X_sz(j,1:6728)); 
    y_test_mat(j-40,1) = X_sz(j,6729);
end

%% Save Data

save_dir = 'C:\Users\soumy\Dropbox (Personal)\Wayne_State_PhD\Lab_work\Connectome_for_Soumyanil_new_classification_data_July17_2019\new_classification_data\original_data\';

% save the augmented train and test expressive data
save(strcat(save_dir,'x_train_density_',density,'.mat'),'x_train_mat','-v7.3');
save(strcat(save_dir,'y_train_density_',density,'.mat'),'y_train_mat');

save(strcat(save_dir,'x_test_density_',density,'.mat'),'x_test_mat');
save(strcat(save_dir,'y_test_density_',density,'.mat'),'y_test_mat');


