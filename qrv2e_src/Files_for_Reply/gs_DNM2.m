function deviance = gs_DNM2(Vcu,maxV,ntrials,params_range,gs)
%%%%%%%%%%%%%%%%%%%%%%%%%
%%% grid search: DNM2 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

%sum of values per trial can be calculated outside the loops
sumVcu = sum(Vcu,2);

%run the grid search loops
deviance = zeros(gs,gs)+log(1/3)*ntrials*-2;

for p1 = 1:gs
    sH = params_range(1,p1);
    for p2 = 1:gs
        w = params_range(2,p2);
        if (sH+w*maxV)>0 %run search only if denominator stays positive
            normalizer = repmat((sH+w*sumVcu),1,3);
            M = Vcu./normalizer;

            P = exp(M(:,1))./sum(exp(M),2);

%             P = min(max(P,.01),.99); %adjustment (to avoid punishing models too much for very unlikely predictions)
            deviance(p1,p2) = -2*sum(log(P));
        end
    end
end

end