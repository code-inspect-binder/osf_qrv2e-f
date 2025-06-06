function deviance = gs_probit_fast(Vcu,ntrials,params_range,gs)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% grid search: random utility (probit) model %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%compute expected firing rates outside of ("noise") loop
K = 100;
sH = 50;
w = 0;
S = 0;
normalizer = repmat((sH+w*(sum(Vcu,2))),1,3);
M = K.*Vcu./normalizer;
        
deviance = zeros(gs,1)+log(1/3)*ntrials*-2;

for x = 1:gs
        sf = params_range(x);
        sf2 = sf.^2;
        
        fun = @(x,o1,o2,o3,sf2,S) normpdf(x,o1,sqrt(S.*o1+(sf2))).*normcdf((x-o2)./sqrt(S.*o2+(sf2))).*normcdf((x-o3)./sqrt(S.*o3+(sf2)));
%         P = integral(@(x)fun(x,M(:,1),M(:,2),M(:,3),sf2,S),-inf,inf,'ArrayValued',true);
        integralbound = ceil(norminv(.999,max(max(M)),sqrt(S.*max(max(M))+sf2)));
        P = integral(@(x)fun(x,M(:,1),M(:,2),M(:,3),sf2,S),-integralbound,integralbound,'ArrayValued',true);
        
        P = min(max(P,.01),.99); %adjustments (to avoid complex numbers due to numerical integration error and to avoid punishing models too much for very unlikely predictions)
        deviance(x) = -2*sum(log(P));
end

end