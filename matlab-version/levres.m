function[carac] = levres(img)

% La fonction prend une image RGB en entrée et retourne sa caractéristique
% pour les levres.
carac = 0;
detect = vision.CascadeObjectDetector('Mouth');
detect.MinSize = [30 55];
detect.MaxSize = [55 90];
bboxes = detect(img(115:200,:));
bboxes(:,2) = bboxes(:,2)+115;
bboxes=boiteInIm(bboxes);
if bboxes ~= [200 200 1 1]
    im_lab = rgb2lab(img);
    boxlevres = bboxes;
    boxlevres(1,1) = boxlevres(1,1)-20;
    boxlevres(1,3) = boxlevres(1,3)+40;
    im_joue = [im_lab(:,1:21,2),im_lab(:,size(im_lab,2)-20:end,2)];
    im_red = im_lab(boxlevres(1,2):boxlevres(1,2)+boxlevres(1,4),boxlevres(1,1):boxlevres(1,1)+boxlevres(1,3),2);
    carac = abs(max(max(im_red))-max(max(im_joue)));
end