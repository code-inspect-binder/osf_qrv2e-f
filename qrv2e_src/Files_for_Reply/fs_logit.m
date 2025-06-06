function [params,deviance,exitflag] = fs_logit(Vcu,params_init,params_minmax,fei)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% fminsearchcon: logit %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    options = optimset('MaxFunEvals',fei(1),'MaxIter',fei(2));%,'Display','iter');
%     options = optimset('Display','iter');
    [params,deviance,exitflag] = fminsearchcon(@find_params,params_init,params_minmax(1,:),params_minmax(2,:),[],[],[],options);
    
    function deviance = find_params(params)
        
        sH = params(1);
        w = 0;

        sumVcu = (sum(Vcu,2));
        normalizer = repmat((sH+w*sumVcu),1,3);
        M = Vcu./normalizer;

        P = exp(M(:,1))./sum(exp(M),2);

%         P = min(max(P,.01),.99); %adjustments (to avoid punishing models too much for very unlikely predictions)
        deviance = -2*sum(log(P));
        
    end

end