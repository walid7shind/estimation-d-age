function[carac] = sillon_naso(img)

% La fonction prend une image RGB en entrée et retourne sa caractéristique
% pour le sillon nasogénien.

detect = vision.CascadeObjectDetector('Nose');
detect.MinSize = [45 35];
detect.MaxSize = [90 80];
bboxes = detect(img(50:150,60:140,:));
bboxes(:,2) = bboxes(:,2)+50;
bboxes(:,1) = bboxes(:,1)+60;
bboxes=boiteInIm(bboxes);
carac = 0;       
if bboxes ~= [200 200 1 1] 
    boxsillon = bboxes;
    boxsillon(1,1) = boxsillon(1,1)-20;
    boxsillon(1,3) = boxsillon(1,3)+40;
    im_lab = rgb2lab(img);
    [grad,~] = imgradientxy(im_lab(boxsillon(1,2):boxsillon(1,2)+boxsillon(1,4),boxsillon(1,1):boxsillon(1,1)+boxsillon(1,3),1));
    im1 = grad(:,1:21);
    im2 = grad(:,size(grad,2)-20:end);
    carac = mean(im1,"all")/2+mean(im2,"all")/2;
end