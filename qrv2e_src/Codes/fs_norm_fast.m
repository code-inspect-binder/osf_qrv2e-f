function [params,deviance,exitflag] = fs_norm_fast(Vcu,params_init,params_minmax,fei)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% fminsearchcon: normalization model %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    options = optimset('MaxFunEvals',fei(1),'MaxIter',fei(2));%,'Display','iter');
%     options = optimset('Display','iter');
    [params,deviance,exitflag] = fminsearchcon(@find_params,params_init,params_minmax(1,:),params_minmax(2,:),[],[],[],options);

    function deviance = find_params(params)
        
        K = params(1);
        sH = params(2);
        w = params(3);
        sf = params(4);
        S = params(5);

        sf2 = sf.^2;
        normalizer = repmat((sH+w*(sum(Vcu,2))),1,3);
        M = K.*Vcu./normalizer;

        fun = @(x,o1,o2,o3,sf2,S) normpdf(x,o1,sqrt(S.*o1+(sf2))).*normcdf((x-o2)./sqrt(S.*o2+(sf2))).*normcdf((x-o3)./sqrt(S.*o3+(sf2)));
        integralbound = ceil(norminv(.999,max(max(M)),sqrt(S.*max(max(M))+sf2)));
        P = integral(@(x)fun(x,M(:,1),M(:,2),M(:,3),sf2,S),-integralbound,integralbound,'ArrayValued',true);
        
        P = min(max(P,.01),.99); %adjustments (to avoid complex numbers due to numerical integration error and to avoid punishing models too much for very unlikely predictions)
        deviance = -2*sum(log(P));
        
    end

end