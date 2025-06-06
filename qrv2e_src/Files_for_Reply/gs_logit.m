function deviance = gs_logit(Vcu,ntrials,params_range,gs)
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% grid search: logit %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%

%sum of values per trial can be calculated outside the loops
sumVcu = sum(Vcu,2);
%fix normalization weight to 0
w = 0;

%run the grid search loops
deviance = zeros(gs,1)+log(1/3)*ntrials*-2;

for p1 = 1:gs
    sH = params_range(p1);
    normalizer = repmat((sH+w*sumVcu),1,3);
    M = Vcu./normalizer;
        
    P = exp(M(:,1))./sum(exp(M),2);
      
%     P = min(max(P,.01),.99); %adjustment (to avoid punishing models too much for very unlikely predictions)
    deviance(p1) = -2*sum(log(P));
end

end