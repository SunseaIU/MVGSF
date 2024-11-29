function [f_v,f_1,f_2,f_3,f_4] = getFunctionValue(beta,lambda,delta,alpha,D_X,S,F,L_s,view_num)
    f_1=0;
    for v=1:view_num
        temp=D_X(:,:,v).*S;
        temp_sum=sum(temp(:));
        f_1=f_1+alpha(v)*temp_sum;
    end
    f_2=beta*norm(S, 'fro')^2;
    f_3=2*lambda*trace(F'*L_s*F);
    f_4=delta*alpha*alpha';
    f_v = f_1+f_2+f_3+f_4;
end

