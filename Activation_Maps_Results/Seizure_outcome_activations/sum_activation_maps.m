
clear all
close all
clc

%% sum class 0 activation maps - top 20

% load('guided_grad_cam_class_0_top_20_test_cases.mat')
% 
% top = size(guided_grad_cam_class_0_top_20,3);
% 
% summed_map = zeros(size(guided_grad_cam_class_0_top_20,1),size(guided_grad_cam_class_0_top_20,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + guided_grad_cam_class_0_top_20(:,:,i);
%     figure,imagesc(guided_grad_cam_class_0_top_20(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(guided_grad_cam_class_0_top_20,3);
% summed_map = summed_map./max(summed_map(:));
% figure, imagesc(summed_map)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% sum class 1 activation maps - top 20

load('guided_grad_cam_class_1_top_20_test_cases.mat')

top = size(guided_grad_cam_class_1_top_20,3);

summed_map = zeros(size(guided_grad_cam_class_1_top_20,1),size(guided_grad_cam_class_1_top_20,2));

for i=1:top
   
    summed_map = summed_map + guided_grad_cam_class_1_top_20(:,:,i);
    figure,imagesc(guided_grad_cam_class_1_top_20(:,:,i))
    
end

summed_map = summed_map./size(guided_grad_cam_class_1_top_20,3);
summed_map = summed_map./max(summed_map(:));
figure, imagesc(summed_map)


