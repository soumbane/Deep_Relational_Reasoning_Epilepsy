%% Convert script for Regression - Fold 1
%% convert matrix of vectors to matrix - EP conn - train - 20951, test - 5110 patients
% Augmented data

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

x_train_expressive_mat = zeros(116,116,20951); % 20951 patients with 511 aug data for each patient (511*41) 
y_train_expressive_mat = zeros(20951,1);

% Form the augmented training set - 41*511 = 20951 training samples
% original connectomes

for i=11:51
    x_train_expressive_mat(:,:,i-10) = vec_to_mat(X_expressive(i,1:6728));   
    y_train_expressive_mat(i-10,1) = X_expressive(i,6729);    
end

% augmented connectomes
start = ((51+510*10)+1); % start of augmented train set

for i=start:26061
    x_train_expressive_mat(:,:,(i-(start-41))+1) = vec_to_mat(X_expressive(i,1:6728));   
    y_train_expressive_mat((i-(start-41))+1,1) = X_expressive(i,6729);    
end

%% receptive score data

x_train_receptive_mat = zeros(116,116,20951); % 20951 patients with 511 aug data for each patient (511*41) 
y_train_receptive_mat = zeros(20951,1);

% Form the augmented training set - 34*511 = 17374 training samples
% original connectomes

for i=11:51
    x_train_receptive_mat(:,:,i-10) = vec_to_mat(X_receptive(i,1:6728));   
    y_train_receptive_mat(i-10,1) = X_receptive(i,6729);    
end

% augmented connectomes
start = ((51+510*10)+1); % start of augmented train set

for i=start:26061
    x_train_receptive_mat(:,:,(i-(start-41))+1) = vec_to_mat(X_receptive(i,1:6728));   
    y_train_receptive_mat((i-(start-41))+1,1) = X_receptive(i,6729);    
end


%% Construct Test Data

%% expressive score data

x_test_expressive_mat = zeros(116,116,5110); % 5110 patients 
y_test_expressive_mat = zeros(5110,1);

% Form the test set - 5110 testing samples

% original connectomes

for j = 1:10
    x_test_expressive_mat(:,:,j) = vec_to_mat(X_expressive(j,1:6728)); 
    y_test_expressive_mat(j,1) = X_expressive(j,6729);
end

% augmented connectomes
end_test = ((51+510*10)); % end of augmented test set

for j = 52:end_test
    x_test_expressive_mat(:,:,j-41) = vec_to_mat(X_expressive(j,1:6728)); 
    y_test_expressive_mat(j-41,1) = X_expressive(j,6729);
end

%% receptive score data

x_test_receptive_mat = zeros(116,116,5110); % 5110 patients 
y_test_receptive_mat = zeros(5110,1);

% Form the test set - 5110 testing samples

% original connectomes

for j = 1:10
    x_test_receptive_mat(:,:,j) = vec_to_mat(X_receptive(j,1:6728)); 
    y_test_receptive_mat(j,1) = X_receptive(j,6729);
end

% augmented connectomes
end_test = ((51+510*10)); % end of augmented test set

for j = 52:end_test
    x_test_receptive_mat(:,:,j-41) = vec_to_mat(X_receptive(j,1:6728)); 
    y_test_receptive_mat(j-41,1) = X_receptive(j,6729);
end

%% Save Data

%% expressive data

save_dir = 'C:\Users\soumy\Dropbox (Personal)\Wayne_State_PhD\Lab_work\Connectome_regression_data\processed_data\expressive_data\fold1\';

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

save_dir = 'C:\Users\soumy\Dropbox (Personal)\Wayne_State_PhD\Lab_work\Connectome_for_Soumyanil_new_data_regression_July1_2019\processed_data\receptive_data\fold1\';

% save the augmented train and test receptive data
save(strcat(save_dir,'x_train_receptive_density_',density,'.mat'),'x_train_receptive_mat','-v7.3');
save(strcat(save_dir,'y_train_receptive_density_',density,'.mat'),'y_train_receptive_mat');

save(strcat(save_dir,'x_test_receptive_density_',density,'.mat'),'x_test_receptive_mat');
save(strcat(save_dir,'y_test_receptive_density_',density,'.mat'),'y_test_receptive_mat');

