
clear all
close all
clc

%% sum expressive activation maps - top 20

% load('grad_ram_expressive_matrix_top_20_test_cases.mat')
% 
% top = size(grad_ram_expressive_top_20,3);
% 
% summed_map = zeros(size(grad_ram_expressive_top_20,1),size(grad_ram_expressive_top_20,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + grad_ram_expressive_top_20(:,:,i);
%     figure,imagesc(grad_ram_expressive_top_20(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(grad_ram_expressive_top_20,3);
% figure, imagesc(summed_map)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% sum receptive activation maps - top 20

% load('grad_ram_receptive_matrix_top_20_test_cases.mat')
% 
% top = size(grad_ram_receptive_top_20,3);
% 
% summed_map = zeros(size(grad_ram_receptive_top_20,1),size(grad_ram_receptive_top_20,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + grad_ram_receptive_top_20(:,:,i);
%     figure,imagesc(grad_ram_receptive_top_20(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(grad_ram_receptive_top_20,3);
% figure, imagesc(summed_map)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% sum expressive activation maps - top 200

% load('grad_ram_expressive_matrix_top_200_test_cases.mat')
% 
% top = size(grad_ram_expressive_top_200,3);
% 
% summed_map = zeros(size(grad_ram_expressive_top_200,1),size(grad_ram_expressive_top_200,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + grad_ram_expressive_top_200(:,:,i);
% %     figure,imagesc(grad_ram_expressive_top_200(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(grad_ram_expressive_top_200,3);
% figure, imagesc(summed_map)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% sum receptive activation maps - top 200

% load('grad_ram_receptive_matrix_top_200_test_cases.mat')
% 
% top = size(grad_ram_receptive_top_200,3);
% 
% summed_map = zeros(size(grad_ram_receptive_top_200,1),size(grad_ram_receptive_top_200,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + grad_ram_receptive_top_200(:,:,i);
% %     figure,imagesc(grad_ram_receptive_top_200(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(grad_ram_receptive_top_200,3);
% figure, imagesc(summed_map)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% guided - sum expressive activation maps - top 20

% load('guided_grad_ram_expressive_matrix_top_20_test_cases.mat')
% 
% top = size(guided_grad_ram_expressive_top_20,3);
% 
% summed_map = zeros(size(guided_grad_ram_expressive_top_20,1),size(guided_grad_ram_expressive_top_20,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + guided_grad_ram_expressive_top_20(:,:,i);
% %     figure,imagesc(guided_grad_ram_expressive_top_20(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(guided_grad_ram_expressive_top_20,3);
% figure, imagesc(summed_map)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% guided - sum receptive activation maps - top 20

% load('guided_grad_ram_receptive_matrix_top_20_test_cases.mat')
% 
% top = size(guided_grad_ram_receptive_top_20,3);
% 
% summed_map = zeros(size(guided_grad_ram_receptive_top_20,1),size(guided_grad_ram_receptive_top_20,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + guided_grad_ram_receptive_top_20(:,:,i);
% %     figure,imagesc(guided_grad_ram_receptive_top_20(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(guided_grad_ram_receptive_top_20,3);
% figure, imagesc(summed_map)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% guided - sum expressive activation maps - top 200

% load('guided_grad_ram_expressive_matrix_top_200_test_cases.mat')
% 
% top = size(guided_grad_ram_expressive_top_200,3);
% 
% summed_map = zeros(size(guided_grad_ram_expressive_top_200,1),size(guided_grad_ram_expressive_top_200,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + guided_grad_ram_expressive_top_200(:,:,i);
%     figure,imagesc(guided_grad_ram_expressive_top_200(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(guided_grad_ram_expressive_top_200,3);
% figure, imagesc(summed_map)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% conv2d_3/conv_2d_2/conv2d_4 - guided - sum expressive activation maps - top 20

% load('guided_conv2d_4_grad_ram_expressive_matrix_top_20_test_cases.mat')
% 
% top = size(guided_conv2d_4_grad_ram_expressive_top_20,3);
% 
% summed_map = zeros(size(guided_conv2d_4_grad_ram_expressive_top_20,1),size(guided_conv2d_4_grad_ram_expressive_top_20,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + guided_conv2d_4_grad_ram_expressive_top_20(:,:,i);
% %     figure,imagesc(guided_conv2d_2_grad_ram_expressive_top_20(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(guided_conv2d_4_grad_ram_expressive_top_20,3);
% figure, imagesc(summed_map)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% conv2d_3/conv2d_2/conv2d_4 - guided - sum receptive activation maps - top 20

% load('guided_conv2d_4_grad_ram_receptive_matrix_top_20_test_cases.mat')
% 
% top = size(guided_conv2d_4_grad_ram_receptive_top_20,3);
% 
% summed_map = zeros(size(guided_conv2d_4_grad_ram_receptive_top_20,1),size(guided_conv2d_4_grad_ram_receptive_top_20,2));
% 
% for i=1:top
%    
%     summed_map = summed_map + guided_conv2d_4_grad_ram_receptive_top_20(:,:,i);
% %     figure,imagesc(guided_conv2d_4_grad_ram_receptive_top_20(:,:,i))
%     
% end
% 
% summed_map = summed_map./size(guided_conv2d_4_grad_ram_receptive_top_20,3);
% figure, imagesc(summed_map)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% sum expressive activation maps - top 300

load('guided_grad_ram_expressive_matrix_top_300_test_cases.mat')

top = size(guided_grad_ram_expressive_top_300,3);

summed_map = zeros(size(guided_grad_ram_expressive_top_300,1),size(guided_grad_ram_expressive_top_300,2));

for i=1:top
   
    summed_map = summed_map + guided_grad_ram_expressive_top_300(:,:,i);
%     figure,imagesc(guided_grad_ram_expressive_top_300(:,:,i))
    
end

summed_map = summed_map./size(guided_grad_ram_expressive_top_300,3);
figure, imagesc(summed_map)
