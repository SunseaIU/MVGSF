function alpha = updateAlpha(D_X,S,n,v,delta)
alpha=zeros(1,v);
h=zeros(1,v);
for p=1:v
    for i = 1:n
        for j=1:n
            h(1,p) = h(p)+D_X(i,j,p)*S(i,j);
        end
    end
end
temp=-h/(2*delta);
alpha= EProjSimplex_new(temp);
end

