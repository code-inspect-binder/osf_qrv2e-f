function deviance = gs_norm_fast(Vcu,ntrials,params_range,gs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% grid search: normalization model %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
deviance = zeros(gs,gs,gs,gs,gs)+log(1/3)*ntrials*-2;

for p1 = 1:gs
    K = params_range(1,p1);
    for p2 = 1:gs
        sH = params_range(2,p2);
        for p3 = 1:gs
            w = params_range(3,p3);
            normalizer = repmat((sH+w*(sum(Vcu,2))),1,3);
            M = K.*Vcu./normalizer; 
            for p4 = 1:gs
                sf = params_range(4,p4);
                sf2 = sf.^2;                
                for p5 = 1:gs
                    S = params_range(5,p5);
        
                    fun = @(x,o1,o2,o3,sf2,S) normpdf(x,o1,sqrt(S.*o1+(sf2))).*normcdf((x-o2)./sqrt(S.*o2+(sf2))).*normcdf((x-o3)./sqrt(S.*o3+(sf2)));
                    integralbound = ceil(norminv(.999,max(max(M)),sqrt(S.*max(max(M))+sf2)));
                    P = integral(@(x)fun(x,M(:,1),M(:,2),M(:,3),sf2,S),-integralbound,integralbound,'ArrayValued',true);

                    P = min(max(P,.01),.99); %adjustments (to avoid complex numbers due to numerical integration error and to avoid punishing models too much for very unlikely predictions)
                    deviance(p1,p2,p3,p4,p5) = -2*sum(log(P));
                end
            end
        end
    end
end
   
end