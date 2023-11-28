function [fighandle] = tvscale (image,fig_title,minmin,maxmax)

%%scales a given image to a desired value
if strcmp(fig_title,'Data')
    
    temp = squeeze(image);
    no_rows = size(temp,1);
    no_cols = size(temp,2);
    
    fighandle = figure;
    set(fighandle,'Position',[10,10,no_rows*10,no_cols*10],'Resize','off');
    
    imagesc(temp,[minmin,maxmax]);
    title('GATE DATA','fontweight','bold','fontsize',20)
    
elseif strcmp(fig_title,'Fit')
    
    temp = squeeze(image);
    no_rows = size(temp,1);
    no_cols = size(temp,2);
    
    fighandle = figure;
    set(fighandle,'Position',[10,10,no_rows*10,no_cols*10],'Resize','off');
    
    imagesc(temp,[minmin,maxmax]);
    title('FITTED MODEL','fontweight','bold','fontsize',20)
    
else
    temp = squeeze(image);
    no_rows = size(temp,1);
    no_cols = size(temp,2);
    
    fighandle = figure;
    set(fighandle,'Position',[10,10,no_rows*10,no_cols*10],'Resize','off');
    
    imagesc(temp,[minmin,maxmax]);
    title('DIFFERENCE','fontweight','bold','fontsize',20)
    
end


end