function[boite] = boiteInIm(bb)

% Cette fonction permet de ne garder qu'une boîte parmmi celles
% détectées par Viola Jones

boite = [200 200 1 1];
[m,~] = size(bb);
for i=1:m
    if (bb(i,1)<200 && bb(i,2)<200) % Boîte dans l'image
        if(bb(i,1)<boite(1,1)) % Boîte la plus haute
            boite = bb(i,:);
        end
    end
end