function[carac] = ride(img)

% La fonction prend une image RGB en entrée et retourne sa caractéristique
% pour les rides.
carac = 0;
detect = vision.CascadeObjectDetector('Mouth');
detect.MinSize = [30 55];
detect.MaxSize = [55 90];
bboxes = detect(img(115:200,:));
bboxes(:,2) = bboxes(:,2)+115;
bboxes=boiteInIm(bboxes);
if bboxes ~= [200 200 1 1] 
    boxride = bboxes;
    boxride(1,1) = boxride(1,1)-20;
    boxride(1,3) = boxride(1,3)+40;
    im_lab = rgb2lab(img);
    [gradx,grady]=imgradientxy(im_lab(boxride(1,2):boxride(1,2)+boxride(1,4),boxride(1,1):boxride(1,1)+boxride(1,3),1));
    grad = sqrt(gradx.*gradx+grady.*grady);
    im2 = grad(:,1:21);
    im3 = grad(:,size(grad,2)-20:end);
    carac = mean(im2,"all")/2+mean(im3,"all")/2;
end