function DivNorm_Analysis_EyeTrack_ReplyToReply
% Divisive Normalization Project
% Experiment: Eye Tracking
% 
% This script runs the following analyses related to the reply to the 
% comment by Webb and colleagues on our manuscript:
% - Model comparison between multinomial Logit and two versions of the
% divisive normalization model (with paramter w being restricted to
% positive values or not) via Maximum Likelihood Estimation (MLE)
% - The same but via hierarchical Bayesian modeling (HBM)
%
% The following additional files are required to run this script:
% - "behavdata_for_Matlab" (behavioral data prepared in R)
% - gs_logit.m (for the MLE-based model comparison)
% - fs_logit.m (for the MLE-based model comparison)
% - gs_DNM.m (for the MLE-based model comparison)
% - fs_DNM.m (for the MLE-based model comparison)
% - gs_DNM2.m (for the MLE-based model comparison)
% - fs_DNM2.m (for the MLE-based model comparison)
% - fminsearchcon.m (for the MLE-based model comparison)
% - 'JAGS_model_DNM_uniform.txt' (for HBM-based model comparison)
% - 'JAGS_model_DNM2_uniform.txt' (for HBM-based model comparison)
% - 'JAGS_model_logit_uniform.txt' (for HBM-based model comparison)
%
% The following tools/libraries are required to run this script
% - matjags (for the hierarchical Bayesian analyses)
% - parallel computing (for the hierarchical Bayesian analyses - but could
%   also be switched off)
%
% sebastian.gluth@unibas.ch


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Load and prepare data %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datapath = pwd; %assume that data is in same folder (adapt if necessary)

%behavior
behavdata = csvread([datapath,'/behavdata_for_Matlab'],1,0);
behavheader = {'item1','item2','item3','distractor','difficulty','rating1','rating2','rating3','position1','position2','position3',...
               'choice','RT','rating1c','rating2c','rating3c','position1c','position2c','position3c','choice123','trialnr','ID'};
%fixations
fixdata = csvread([datapath,'/fixdata_for_Matlab'],1,0);
fixheader = {'fixdur','intrialonset','AOI','trialnr','ID'};
isfirstfix = [1;fixdata(2:end,ismember(fixheader,'trialnr'))~=fixdata(1:end-1,ismember(fixheader,'trialnr'))]==1;
% islastfix = [fixdata(2:end,ismember(fixheader,'trialnr'))~=fixdata(1:end-1,ismember(fixheader,'trialnr'));1]==1;

IDs = unique(behavdata(:,ismember(behavheader,'ID')));
nsubj = length(IDs);

fxdata.V3 = []; %will be dataset for fixed effects analysis
fxdata.normV3 = []; %will be dataset for fixed effects analysis
%within-subject analyses
for s = 1:nsubj
    
    %preparations
    cdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    
    V3 = cdata(:,ismember(behavheader,'rating3c')); %value of worst option (distractor)
    V2 = cdata(:,ismember(behavheader,'rating2c')); %value of 2nd best option (target 2)
    V1 = cdata(:,ismember(behavheader,'rating1c')); %value of best option (target 1)
    normV3 = 2*V3./(V1+V2); %value of distractor "normalized" to value of targets
    
    %descriptive statistics
    descriptives.best_chosen(s) = mean(cdata(:,ismember(behavheader,'choice123'))==1);
    descriptives.second_best_chosen(s) = mean(cdata(:,ismember(behavheader,'choice123'))==2);
    descriptives.distractor_chosen(s) = mean(cdata(:,ismember(behavheader,'choice123'))==3);
    descriptives.RT(s) = mean(cdata(:,ismember(behavheader,'RT')));
    
    %clean up data: exclude any trials in which targets were not chosen
    targetchosen = cdata(:,ismember(behavheader,'choice123'))<=2;
    V1c = V1(targetchosen);
    V2c = V2(targetchosen);
    V3c = V3(targetchosen);
    normV3c = normV3(targetchosen);
    Choice = cdata(targetchosen,ismember(behavheader,'choice123'))==1; %1 = best chosen, 0 = 2nd best chosen
    
    %analysis 1a: choice efficiency as a function of high vs. low V3
    V3chigh = V3c>median(V3c);
    analysis.a1(s,:) = [mean(Choice(V3chigh==1)),mean(Choice(V3chigh==0))];
    %analysis 1b: choice efficiency as a function of high vs. low normV3
    normV3chigh =  normV3c>median(normV3c);
    analysis.b1(s,:) = [mean(Choice(normV3chigh==1)),mean(Choice(normV3chigh==0))];
    
    %analysis 2a: multiple regression with V3
    X = [V1c,V2c,V3c]; %independent variables
    Xz = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1); %standardize independent variables
    analysis.a2.betas(s,:) = glmfit(Xz,Choice,'binomial');
    fxdata.V3 = [fxdata.V3;[X,Choice,zeros(length(X),1)+s]]; %prepare analysis 3a (fixed effects multiple regression)
    %analysis 2b: multiple regression with normV3
    X = [V1c,V2c,normV3c]; %independent variables
    Xz = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1); %standardize independent variables
    analysis.b2.betas(s,:) = glmfit(Xz,Choice,'binomial');
    fxdata.normV3 = [fxdata.normV3;[X,Choice,zeros(length(X),1)+s]]; %prepare analysis 3b (fixed effects multiple regression)
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Frequentist statistics %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%exclude no (additional) subjects (done in R or even prior to R)
inclsubj = 1:nsubj; %for random effects analyses
incltrials = fxdata.V3(:,5)>0; %for fixed effects analyses

%analysis 1a and 1b on the group level
%(choice efficiency as a function of high vs. low V3/normV3)
[~,p,~,t] = ttest(analysis.a1(inclsubj,1),analysis.a1(inclsubj,2),'tail','left');
[~,~,ci,~] = ttest(analysis.a1(inclsubj,:));
d = mean(analysis.a1(inclsubj,1)-analysis.a1(inclsubj,2))./std(analysis.a1(inclsubj,1)-analysis.a1(inclsubj,2));
disp(['t-test of high vs. low V3: t(',int2str(t.df),') = ',num2str(t.tstat,3),'; p = ',num2str(p,3),'; d = ',num2str(d,3)])

figure;hold on;title('Eye-tracking experiment');ylabel('Relative choice accuracy');
xlim([0.5,2.5]);ylim([0.25,0.95])
plot([1,1],ci(:,1),'k-','LineWidth',10);plot([2,2],ci(:,2),'k-','LineWidth',10)
plot(ones(length(inclsubj),1),analysis.a1(inclsubj,1),'.','Color',[.5,.5,.5]);
plot(ones(length(inclsubj),1)+1,analysis.a1(inclsubj,2),'.','Color',[.5,.5,.5])
plot([0.8,1.2],zeros(1,2)+mean(analysis.a1(inclsubj,1)),'-','LineWidth',2,'Color',[.25,1,.25])
plot([1.8,2.2],zeros(1,2)+mean(analysis.a1(inclsubj,2)),'m-','LineWidth',2,'Color',[0,.5,0])
set(gca,'XTick',1:2,'XTickLabel',{'high V3','low V3'},'YTick',.3:.2:.9,'FontSize',12)
% set(gcf,'PaperPosition',[1/4,1/4,8,8]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_LowHigh_V3_R1')

[~,p,~,t] = ttest(analysis.b1(inclsubj,1),analysis.b1(inclsubj,2),'tail','left');
% [~,~,ci,~] = ttest(analysis.b1(inclsubj,:));
d = mean(analysis.b1(inclsubj,1)-analysis.b1(inclsubj,2))./std(analysis.b1(inclsubj,1)-analysis.b1(inclsubj,2));
disp(['t-test of high vs. low normV3: t(',int2str(t.df),') = ',num2str(t.tstat,3),'; p = ',num2str(p,3),'; d = ',num2str(d,3)])

%analysis 2a and 2b on the group level
%(multiple logistic random-effects regression with V3/normV3)
[~,p(2)] = ttest(analysis.a2.betas(inclsubj,2),zeros(size(analysis.a2.betas(inclsubj,2))),'tail','right');
[~,p(3:4)] = ttest(analysis.a2.betas(inclsubj,3:4),zeros(size(analysis.a2.betas(inclsubj,3:4))),'tail','left');
[~,~,ci,t] = ttest(analysis.a2.betas(inclsubj,:));
d = mean(analysis.a2.betas(inclsubj,:))./std(analysis.a2.betas(inclsubj,:));
disp(['t-test of beta(V3): t(',int2str(t.df(4)),') = ',num2str(t.tstat(4),3),'; p = ',num2str(p(4),3),'; d = ',num2str(d(4),3)])

figure;hold on;title('Eye-tracking experiment');ylabel('Effect on relative choice accuracy')
xlim([0.5,3.5]);ylim([-1.5,1.5]);
plot([0.5,3.5],[0,0],'k-')
plot([1,1],ci(:,2),'k-','LineWidth',10);plot([2,2],ci(:,3),'k-','LineWidth',10);plot([3,3],ci(:,4),'k-','LineWidth',10);
% plot(zeros(length(inclsubj),1)+1,analysis.a2.betas(inclsubj,2),'.','Color',[.5,.5,.5]);
% plot(zeros(length(inclsubj),1)+2,analysis.a2.betas(inclsubj,3),'.','Color',[.5,.5,.5]);
plot(zeros(length(inclsubj),1)+3,analysis.a2.betas(inclsubj,4),'.','Color',[.5,.5,.5]);
plot([0.8,1.2],zeros(1,2)+mean(analysis.a2.betas(inclsubj,2)),'b-','LineWidth',2)
plot([1.8,2.2],zeros(1,2)+mean(analysis.a2.betas(inclsubj,3)),'r-','LineWidth',2)
plot([2.8,3.2],zeros(1,2)+mean(analysis.a2.betas(inclsubj,4)),'g-','LineWidth',2)
set(gca,'XTick',1:3,'XTickLabel',{'V1','V2','V3'},'FontSize',12)
% set(gcf,'PaperPosition',[1/4,1/4,8,8]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_LogReg_V3_R1')

[~,p,~,t] = ttest(analysis.b2.betas(inclsubj,:));
d = mean(analysis.b2.betas(inclsubj,:))./std(analysis.b2.betas(inclsubj,:));
disp(['t-test of beta(normV3): t(',int2str(t.df(4)),') = ',num2str(t.tstat(4),3),'; p = ',num2str(p(4),3),'; d = ',num2str(d(4),3)])

%analysis 3a and 3b
%(multiple logistic fixed-effects regression with V3/normV3)
[analysis.a3.B,analysis.a3.DEV,analysis.a3.STATS] = glmfit(fxdata.V3(incltrials,1:3),fxdata.V3(incltrials,4),'binomial');
[analysis.b3.B,analysis.b3.DEV,analysis.b3.STATS] = glmfit(fxdata.normV3(incltrials,1:3),fxdata.normV3(incltrials,4),'binomial');

%pre-analysis for response times: identify cut-off for RT (> mean + 4*SD)
allRT = behavdata(:,ismember(behavheader,'RT'));
RTcut = mean(allRT)+4*std(allRT); %note: min RT is 349 ms, so that's fine

%perform analysis of RT
for s = 1:nsubj %note: the excluded subject is still analyzed, so make sure to exclude her when running the statistics
    
	%preparations
    cdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    
    V3 = cdata(:,ismember(behavheader,'rating3c')); %value of worst option (distractor)
    V2 = cdata(:,ismember(behavheader,'rating2c')); %value of 2nd best option (target 2)
    V1 = cdata(:,ismember(behavheader,'rating1c')); %value of best option (target 1)
    normV3 = 2*V3./(V1+V2); %value of distractor "normalized" to value of targets
    
    %clean up data: exclude any trials in which targets were not chosen
    include_trials = (cdata(:,ismember(behavheader,'choice123'))<=2)&(cdata(:,ismember(behavheader,'RT'))<=RTcut);
    V1c = V1(include_trials);
    V2c = V2(include_trials);
    V3c = V3(include_trials);
    normV3c = normV3(include_trials);
    RT = cdata(include_trials,ismember(behavheader,'RT'));
        
    %multiple regression of RT with V3
    X = [V1c-V2c,V1c+V2c,V3c]; %independent variables
    Xz = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1); %standardize independent variables
    analysis.RTa.betas(s,:) = glmfit(Xz,RT); %analysis of RT
    %analysis 2b: multiple regression with normV3
    X = [V1c-V2c,V1c+V2c,normV3c]; %independent variables
    Xz = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1); %standardize independent variables
    analysis.RTb.betas(s,:) = glmfit(Xz,RT); %analysis of RT
    
    %plot RT as a function of V3 (similar to Fig. 5A in Louie et al., 2013)
    ncatRT = 5;
    V3cat = sortrows([sortrows([(1:length(V3))',V3],2),sort(repmat((1:ncatRT)',length(V3)/(ncatRT),1))],1)*[0;0;1];
    V3ccat = V3cat(include_trials);
    for c = 1:ncatRT
        ct = V3ccat==c; %trials belonging to current category
        analysis.RTb.visualization_V3(s,c) = mean(RT(ct));
    end
    
end

%figure for RT effects
[~,~,ci,~] = ttest(analysis.RTa.betas(inclsubj,2:end));
figure('OuterPosition',[440 378 600 800]);hold on;title('Eye-tracking experiment');ylabel('Effect on response time')
xlim([0.5,3.5]);ylim([-1000,500])
plot([0.5,3.5],[0,0],'k-')
for x = 1:size(ci,2)
    plot([x,x],ci(:,x),'k-','LineWidth',10);
    plot(zeros(length(inclsubj),1)+x,analysis.RTa.betas(inclsubj,x+1),'.','Color',[.5,.5,.5])
end
plot([0.8,1.2],zeros(1,2)+nanmean(analysis.RTa.betas(inclsubj,2)),'b-','LineWidth',2)
plot([1.8,2.2],zeros(1,2)+nanmean(analysis.RTa.betas(inclsubj,3)),'r-','LineWidth',2)
plot([2.8,3.2],zeros(1,2)+nanmean(analysis.RTa.betas(inclsubj,4)),'g-','LineWidth',2)
set(gca,'XTick',1:3,'XTickLabel',{'V1 - V2','V1 + V2','V3'},'FontSize',16)
% set(gcf,'PaperPosition',[1/4,1/4,12,12]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_LinReg_RT_R1')

%plot RT similar Figure 5A
[~,~,ci,~] = ttest(analysis.RTb.visualization_V3(inclsubj,:));
figure;hold on;xlim([0.5,5.5]);ylim([1550,2300])
for x = 1:ncatRT
    plot([x,x],ci(:,x),'k-','LineWidth',6);
    plot([0.8,1.2]+(x-1),zeros(1,2)+nanmean(analysis.RTb.visualization_V3(inclsubj,x)),'g-','LineWidth',3)
end
plot(nanmean(analysis.RTb.visualization_V3(inclsubj,:)),'g-')
set(gca,'XTick',[1,5],'XTickLabel',{'lowest','highest'},'YTick',1600:200:2200,'FontSize',16)
xlabel('V3');ylabel('RT in ms')
% set(gcf,'PaperPosition',[1/4,1/4,12,9]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_RTbyV3')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Maximum Likelihood Estimation %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all

%general settings
fei = [2000,1000]; %max number of function evaluations and iterations
fs = [1,3,5,1]; %number of fminsearches with best parameter estimates from grid search; random pick of best 30%/1%; random pick; nested (or random again)

%logit model settings
logit.gs = 10; %number of gridsearches per parameter for logit model
logit.params_minmax = [eps;20]; %the free parameter of this model is the semisaturation, which basically serves as a scaling of the Gumbel-distributed noise
logit.params_range = linspace(0.01,18.1,logit.gs); %grid search parameter range for logit model

%DNM settings
DNM.gs = [logit.gs,logit.gs]; %number of gridsearches per parameter for DNM model
DNM.params_minmax = [eps,0;20,10]; %parameters are semisaturation and weight
DNM.params_range = [linspace(0.01,18.1,logit.gs);linspace(0,1.81,logit.gs)]; %grid search parameter range for logit model

%DNM2 settings (allows parameter w to be negative)
DNM2.gs = [logit.gs,logit.gs]; %number of gridsearches per parameter for DNM model
DNM2.params_minmax = [eps,-5;20,5]; %parameters are semisaturation and weight
DNM2.params_range = [linspace(0.01,18.1,logit.gs);linspace(-0.9,0.9,logit.gs)]; %grid search parameter range for logit model

for s = 1:nsubj
%     tic
    %preparations
    cdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    
    V3 = cdata(:,ismember(behavheader,'rating3c')); %value of worst option (distractor)
    V2 = cdata(:,ismember(behavheader,'rating2c')); %value of 2nd best option (target 2)
    V1 = cdata(:,ismember(behavheader,'rating1c')); %value of best option (target 1)
    
    C = cdata(:,ismember(behavheader,'choice123')); %1 = best chosen, 2 = 2nd best chosen, 3 = worst chosen
    Vcu = [V3.*(C==3)+V2.*(C==2)+V1.*(C==1),V2.*(C==1)+V1.*(C==2)+V1.*(C==3),V3.*(C==1)+V3.*(C==2)+V2.*(C==3)]; %value of chosen, better unchosen and worse unchosen option
    ntrials = length(C);
    
    %logit model, grid search
    deviance = gs_logit(Vcu,ntrials,logit.params_range,logit.gs);
    logit.dev_gs(s,:) = deviance;
    %fminsearch
    preparams = zeros(sum(fs),1);
	predev = zeros(sum(fs),1);
	best_gs = sortrows([deviance,(1:logit.gs)'])*[0;1];
    for x = 1:sum(fs)
        if x <= fs(1) %start with best
            params_init = logit.params_range(best_gs(1));
        elseif (x > fs(1)) && (x <= sum(fs(1:2))) %start with good (10%)
            params_init = logit.params_range(best_gs(randi(round(logit.gs/10))+1));
        else %start with random
            params_init = logit.params_range(randi(logit.gs));
        end
        [preparams(x),predev(x)] = fs_logit(Vcu,params_init,logit.params_minmax,fei);
    end
    logit.params(s) = preparams(find(predev==min(predev),1));
    logit.dev_fs(s) = predev(find(predev==min(predev),1));
    
	%DNM, grid search
    deviance = gs_DNM(Vcu,ntrials,DNM.params_range,DNM.gs(1));
    DNM.dev_gs{s} = deviance;
	%fminsearch
    preparams = zeros(sum(fs),length(DNM.params_minmax));
	predev = zeros(sum(fs),1);
	preexitflag = zeros(sum(fs),1);
    thresh_deviance = max(((1:prod(DNM.gs))==round(prod(DNM.gs)*.01)).*sort(reshape(deviance,prod(DNM.gs),1))');
	[D1,D2] = ind2sub(size(deviance),find(deviance<=thresh_deviance)); %best 1%
	[D1b,D2b] = ind2sub(size(deviance),find(deviance==min(min(deviance)))); %best
    for x = 1:sum(fs)
        if x <= fs(1) %start with best
            r = randi(length(D1b)); %in case of multiple best values
            params_init = [DNM.params_range(1,D1b(r)),DNM.params_range(2,D2b(r))];
        elseif (x > fs(1)) && (x <= sum(fs(1:2))) %start with good (1%)
            r = randi(length(D1));
            params_init = [DNM.params_range(1,D1(r)),DNM.params_range(2,D2(r))];
        elseif (x > sum(fs(1:2))) && (x <= sum(fs(1:3))) %start with random
            params_init = [DNM.params_range(1,randi(DNM.gs(1))),DNM.params_range(2,randi(DNM.gs(2)))];
        else %start with best of logit
            params_init = [logit.params(s),0];
        end
        [preparams(x,:),predev(x),preexitflag(x)] = fs_DNM(Vcu,params_init,DNM.params_minmax,fei);
    end
	DNM.params(s,:) = preparams(find(predev==min(predev),1),:);
    DNM.dev_fs(s) = predev(find(predev==min(predev),1));
	DNM.exitflag(s) = preexitflag(find(predev==min(predev),1));
	DNM.allexitflags(s,:) = preexitflag;
    
    %DNM2, grid search
    maxV = max(sum(Vcu,2)); %get the max sum of values to constrain parameter search
    deviance = gs_DNM2(Vcu,maxV,ntrials,DNM2.params_range,DNM2.gs(1));
    DNM2.dev_gs{s} = deviance;
	%fminsearch
    preparams = zeros(sum(fs),length(DNM2.params_minmax));
	predev = zeros(sum(fs),1);
	preexitflag = zeros(sum(fs),1);
    thresh_deviance = max(((1:prod(DNM2.gs))==round(prod(DNM2.gs)*.01)).*sort(reshape(deviance,prod(DNM2.gs),1))');
	[D1,D2] = ind2sub(size(deviance),find(deviance<=thresh_deviance)); %best 1%
	[D1b,D2b] = ind2sub(size(deviance),find(deviance==min(min(deviance)))); %best
    for x = 1:sum(fs)
        if x <= fs(1) %start with best
            r = randi(length(D1b)); %in case of multiple best values
            params_init = [DNM2.params_range(1,D1b(r)),DNM2.params_range(2,D2b(r))];
        elseif (x > fs(1)) && (x <= sum(fs(1:2))) %start with good (1%)
            r = randi(length(D1));
            params_init = [DNM2.params_range(1,D1(r)),DNM2.params_range(2,D2(r))];
        elseif (x > sum(fs(1:2))) && (x <= sum(fs(1:3))) %start with random (but within constraints)
            params_ok = 0;
            while params_ok == 0
                params_init = [DNM2.params_range(1,randi(DNM2.gs(1))),DNM2.params_range(2,randi(DNM2.gs(2)))];
                params_ok = ([1,maxV]*params_init')>=.000001;
            end
        else %start with best of logit
            params_init = [logit.params(s),0];
        end
        [preparams(x,:),predev(x),preexitflag(x)] = fs_DNM2(Vcu,maxV,params_init,DNM2.params_minmax,fei);
    end
	DNM2.params(s,:) = preparams(find(predev==min(predev),1),:);
    DNM2.dev_fs(s) = predev(find(predev==min(predev),1));
	DNM2.exitflag(s) = preexitflag(find(predev==min(predev),1));
	DNM2.allexitflags(s,:) = preexitflag;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Revision of MA article %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%figure showing the predictions of the DNM2 model (as in figure of the MA
%%%commentary by Webb and colleagues)

%1. get average values of V1 and V2
avgV12 = zeros(nsubj,2);
for s = 1:nsubj
    cdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    V2 = cdata(:,ismember(behavheader,'rating2c')); %value of 2nd best option (target 2)
    V1 = cdata(:,ismember(behavheader,'rating1c')); %value of best option (target 1)
    avgV12(s,:) = [mean(V1),mean(V2)];
end

%2. get predictions of the DNM2 model
nV3 = 21;
V3 = linspace(0,mean(avgV12(:,2)),nV3)';
V123 = [repmat(mean(avgV12),nV3,1),V3];
P1 = zeros(s,nV3);
P2 = zeros(s,nV3);
for s = 1:length(inclsubj)
    sH = DNM2.params(s,1);
    w = DNM2.params(s,2);
    normalizer = repmat((sH+w*sum(V123,2)),1,3);
    M = V123./normalizer;
    P1(s,:) = (exp(M(:,1))./sum(exp(M),2));
    P2(s,:) = (exp(M(:,2))./sum(exp(M),2));
    P1(s,normalizer(:,1)<0) = NaN; %necessary because the DNM2 makes weird predictions if the denominator isn't positive
    P2(s,normalizer(:,1)<0) = NaN; %necessary because the DNM2 makes weird predictions if the denominator isn't positive
end

%4. draw the figure
ABV = 0.65; %area base value
figure('OuterPosition',[440 378 1920/2 1080/2]);hold on;xlim([-0.05,mean(avgV12(:,1))+.05]);ylim([ABV,1])
[~,~,CI] = ttest(P1./(P1+P2));
area(V3,[CI(1,:);CI(2,:)-CI(1,:)]',ABV,'FaceColor',[1,0,1],'EdgeColor','w','FaceAlpha',0.5)
area(V3,CI(1,:)',ABV,'FaceColor','w','EdgeColor','w')
cs = DNM2.params(:,2)<0;
plot(V3,nanmean(P1(cs,:)./(P1(cs,:)+P2(cs,:))),'r.-','LineWidth',2)
[~,~,CI] = ttest(P1(cs,:)./(P1(cs,:)+P2(cs,:)));
area(V3,[CI(1,:);CI(2,:)-CI(1,:)]',ABV,'FaceColor','r','EdgeColor','w','FaceAlpha',0.5)
area(V3,CI(1,:)',ABV,'FaceColor','w','EdgeColor','w')
[~,~,CI] = ttest(P1./(P1+P2));
area(V3,[CI(1,:);CI(2,:)-CI(1,:)]',ABV,'FaceColor',[1,0,1],'EdgeColor','w','FaceAlpha',0.5)
area(V3,CI(1,:)',ABV,'FaceColor','w','EdgeColor','w')
cs = DNM2.params(:,2)>0;
[~,~,CI] = ttest(P1(cs,:)./(P1(cs,:)+P2(cs,:)));
area(V3,[CI(1,:);CI(2,:)-CI(1,:)]',ABV,'FaceColor','b','EdgeColor','w','FaceAlpha',0.3)
area(V3,CI(1,:)',ABV,'FaceColor','w','EdgeColor','w')

plot([mean(avgV12(:,1)),mean(avgV12(:,1))],[0.65,1],'k--')
text(mean(avgV12(:,1))-.1,0.95,'V1')
plot([mean(avgV12(:,2)),mean(avgV12(:,2))],[0.65,1],'k--')
text(mean(avgV12(:,2))+.05,0.95,'V2')

p1 = plot(V3,nanmean(P1./(P1+P2)),'.-','Color',[1,0,1],'LineWidth',2,'MarkerSize',10);
cs = DNM2.params(:,2)>0;
p2 = plot(V3,nanmean(P1(cs,:)./(P1(cs,:)+P2(cs,:))),'b.-','LineWidth',2,'MarkerSize',10);
cs = DNM2.params(:,2)<0;
p3 = plot(V3,nanmean(P1(cs,:)./(P1(cs,:)+P2(cs,:))),'r.-','LineWidth',2,'MarkerSize',10);

legend([p1,p2,p3],{'All participants','Participants with \omega > 0','Participants with \omega < 0'},'Location','NorthWest')
legend boxoff
xlabel('V3');ylabel('Relative choice accuracy')
set(gca,'FontSize',14,'XTick',0:0.5:3,'YTick',0.6:.1:1)
% set(gcf,'PaperPosition',[1/4,1/4,24,9]);print(gcf,'-dtiff','-r300','ReplyToReply_EyeTrack_Figure_V3Predictions')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Hierarchical Bayesian Estimation %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
addpath /Applications/MATLAB_ADD/matjags

% JAGS settings
doparallel = 1;
nchains = 4;
nburnin = 2000;
nsamples = 10000;
thin = 10;

%start parallel pool
p = gcp;
if isempty(p)
    parpool(4);
end

%specify data inputs for JAGS
jagsdata.V1 = [];
jagsdata.V2 = [];
jagsdata.V3 = [];
jagsdata.C = [];
jagsdata.S = [];
jagsdata.maxV = [];
for s = 1:length(inclsubj)

    %preparations
    cdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    
    V3 = cdata(:,ismember(behavheader,'rating3c')); %value of worst option (distractor)
    V2 = cdata(:,ismember(behavheader,'rating2c')); %value of 2nd best option (target 2)
    V1 = cdata(:,ismember(behavheader,'rating1c')); %value of best option (target 1)
    C = cdata(:,ismember(behavheader,'choice123')); %1 = best chosen, 2 = 2nd best chosen, 3 = worst chosen

    %clean up data: exclude any trials in which targets were not chosen
    jagsdata.V1 = [jagsdata.V1;V1];
    jagsdata.V2 = [jagsdata.V2;V2];
    jagsdata.V3 = [jagsdata.V3;V3];
    jagsdata.C = [jagsdata.C;C]; %1 = best chosen, 2 = 2nd best chosen, 3 = worst chosen
    jagsdata.S = [jagsdata.S;zeros(size(V1))+s];
    jagsdata.maxV = [jagsdata.maxV;max(V1+V2+V3)];
end

% Assign Matlab Variables to the Observed Nodes
N = size(jagsdata.C,1);
% subj = inclsubj;
subj = 1:length(inclsubj);

%%%
%%% Divisive Normalization Model
%%%

jags_data = struct('N',N,'subj',subj,'nsubj',length(subj),...
                   'S',jagsdata.S,'C',jagsdata.C,'V1',jagsdata.V1,'V2',jagsdata.V2,'V3',jagsdata.V3,...
                   'sumV',jagsdata.V1+jagsdata.V2+jagsdata.V3);
               
% Initial Values
clear init0
for iv = 1:nchains
    I.r_sH = rand*100;
    I.l_sH = rand*100;
    I.r_w = rand*100;
    I.l_w = rand*100;
    init0(iv) = I;
end

% Use JAGS to Sample
if size(dir('tmpjags'),1)>0
    rmdir('tmpjags','s')
end
tic
fprintf( 'Running JAGS ...\n' );
[samples_DNM, stats_DNM] =  matjags(jags_data,fullfile(pwd,'JAGS_model_DNM_uniform.txt'),init0, ...
	'doparallel' , doparallel,'nchains', nchains,'nburnin', nburnin,'nsamples', nsamples,'thin',thin,...
	'monitorparams',{'r_sH','l_sH','r_w','l_w','sH','w'},...
	'savejagsoutput',1,'verbosity',1,'cleanup',0,'workingdir','tmpjags');
toc

%check that all R-hats are below 1.01
allRhats = [stats_DNM.Rhat.r_sH;stats_DNM.Rhat.l_sH;stats_DNM.Rhat.r_w;stats_DNM.Rhat.l_w;...
            stats_DNM.Rhat.sH;stats_DNM.Rhat.w;...
            stats_DNM.Rhat.deviance];
if max(allRhats) < 1.01
    disp('all R-hats are below 1.01')
else
    disp('WARNING: not all R-hats are below 1.01!')
end

%%%
%%% Do the same but allow w to take on negative values
%%%

% Assign Matlab Variables to the Observed Nodes
jags_data = struct('N',N,'subj',subj,'nsubj',length(subj),...
                   'S',jagsdata.S,'C',jagsdata.C,'V1',jagsdata.V1,'V2',jagsdata.V2,'V3',jagsdata.V3,...
                   'sumV',jagsdata.V1+jagsdata.V2+jagsdata.V3,'maxV',jagsdata.maxV);
               
% Initial Values
clear init0
for iv = 1:nchains
    I.r_sH = rand*100;
    I.l_sH = rand*100;
    I.r_w = rand*100;
    I.l_w = rand*100;
    init0(iv) = I;
end

% Use JAGS to Sample
if size(dir('tmpjags'),1)>0
    rmdir('tmpjags','s')
end
tic
fprintf( 'Running JAGS ...\n' );
[samples_DNM2, stats_DNM2] =  matjags(jags_data,fullfile(pwd,'JAGS_model_DNM2_uniform.txt'),init0, ...
	'doparallel' , doparallel,'nchains', nchains,'nburnin', nburnin,'nsamples', nsamples,'thin',thin,...
	'monitorparams',{'r_sH','l_sH','r_w','l_w','sH','w'},...
	'savejagsoutput',1,'verbosity',1,'cleanup',0,'workingdir','tmpjags');
toc

%check that all R-hats are below 1.01
allRhats = [stats_DNM2.Rhat.r_sH;stats_DNM2.Rhat.l_sH;stats_DNM2.Rhat.r_w;stats_DNM2.Rhat.l_w;...
            stats_DNM2.Rhat.sH;stats_DNM2.Rhat.w;...
            stats_DNM2.Rhat.deviance];
if max(allRhats) < 1.01
    disp('all R-hats are below 1.01')
else
    disp('WARNING: not all R-hats are below 1.01!')
end

%%%
%%% Do the same but fix w to 0 (i.e., multinomial Logit model)
%%%

% Assign Matlab Variables to the Observed Nodes
jags_data = struct('N',N,'subj',subj,'nsubj',length(subj),...
                   'S',jagsdata.S,'C',jagsdata.C,'V1',jagsdata.V1,'V2',jagsdata.V2,'V3',jagsdata.V3);
               
% Initial Values
clear init0 I
for iv = 1:nchains
    I.r_sH = rand*100;
    I.l_sH = rand*100;
    init0(iv) = I;
end

% Use JAGS to Sample
if size(dir('tmpjags'),1)>0
    rmdir('tmpjags','s')
end
tic
fprintf( 'Running JAGS ...\n' );
[samples_logit, stats_logit] =  matjags(jags_data,fullfile(pwd,'JAGS_model_logit_uniform.txt'),init0, ...
	'doparallel' , doparallel,'nchains', nchains,'nburnin', nburnin,'nsamples', nsamples,'thin',thin,...
	'monitorparams',{'r_sH','l_sH','sH'},...
	'savejagsoutput',1,'verbosity',1,'cleanup',0,'workingdir','tmpjags');
toc

%check that all R-hats are below 1.01
allRhats = [stats_logit.Rhat.r_sH;stats_logit.Rhat.l_sH;...
            stats_logit.Rhat.sH;...
            stats_logit.Rhat.deviance];
if max(allRhats) < 1.01
    disp('all R-hats are below 1.01')
else
    disp('WARNING: not all R-hats are below 1.01!')
end

%%%some figures
figure;
subplot(1,2,1);hold on;xlim([-0.3,0.3]);ylim([0,32]);title('Maximum likelihood estimation')
histogram(DNM.params(:,2),-2.1:.01:1.2,'FaceColor','b','EdgeColor','b','FaceAlpha',1);
histogram(DNM2.params(:,2),-2.1:.01:1.2,'FaceColor','r','EdgeColor','r','FaceAlpha',1);
% h1 = histogram(DNM.params(:,2),-2.1:.05:1.2,'FaceColor','b','EdgeColor','b','FaceAlpha',1);
% h2 = histogram(DNM2.params(:,2),-2.1:.05:1.2,'FaceColor','r','EdgeColor','r','FaceAlpha',1);
histogram(DNM.params(:,2),-2.1:.01:1.2,'FaceColor','b','FaceAlpha',0.5,'EdgeColor','b');
plot([0,0],[0,90],'k--','LineWidth',2)
% legend([h1,h2],{'w restricted to ? 0','w allowed to be < 0'});legend boxoff
xlabel('Parameter estimate of \omega');ylabel('Number of participants')
set(gca,'FontSize',14,'YTick',0:20:100)

subplot(1,2,2);hold on;xlim([-0.3,0.3]);ylim([0,32]);title('Hierarchical Bayesian modeling')
histogram(stats_DNM.mean.w,-0.7:.01:0.6,'FaceColor','b','EdgeColor','b','FaceAlpha',1);
histogram(stats_DNM2.mean.w,-0.7:.01:0.6,'FaceColor','r','EdgeColor','r','FaceAlpha',1);
% h1 = histogram(stats_DNM.mean.w,-0.7:.02:0.6,'FaceColor','b','EdgeColor','b','FaceAlpha',1);
% h2 = histogram(stats_DNM2.mean.w,-0.7:.02:0.6,'FaceColor','r','EdgeColor','r','FaceAlpha',1);
histogram(stats_DNM.mean.w,-0.7:.01:0.6,'FaceColor','b','FaceAlpha',0.5,'EdgeColor','b');
plot([0,0],[0,90],'k--','LineWidth',2)
% legend([h1,h2],{'w restricted to ? 0','w allowed to be < 0'});legend boxoff
xlabel('Parameter estimate of \omega');ylabel('Number of participants')
set(gca,'FontSize',14,'YTick',0:20:100)

% set(gcf,'PaperPosition',[1/4,1/4,24,9]);print(gcf,'-dtiff','-r300','ReplyToReply_EyeTrack_Figure_ParameterEstimates')

figure('OuterPosition',[440 378 1920/2 1080/2]);hold on;ylim([0,1.01*max(logit.dev_fs-DNM2.dev_fs)])
% b2 = bar(sort(logit.dev_fs-DNM2.dev_fs,'descend'),'FaceColor','r');
% b1 = bar(sort(logit.dev_fs-DNM.dev_fs,'descend'),'FaceColor','b');
% bar(sort(logit.dev_fs-DNM.dev_fs,'descend'),'FaceColor','b');
b1 = bar((1:nsubj)-.24,(logit.dev_fs-DNM.dev_fs),'FaceColor','b','EdgeColor','b','BarWidth',.45);
b2 = bar((1:nsubj)+.24,logit.dev_fs-DNM2.dev_fs,'FaceColor','r','EdgeColor','r','BarWidth',.45);
% bar(logit.dev_fs-DNM.dev_fs,'FaceColor','b','EdgeAlpha',0);
plot([0,length(inclsubj)+1],zeros(1,2)+chi2inv(.95,1),'k-');
% text(length(inclsubj)*.9,chi2inv(.95,1)*.9,'{\it p} < .05')
xlabel('Participants');ylabel('Improvement of deviance','FontSize',14)
set(gca,'FontSize',14,'XTick',0:10:100,'YTick',0:10:100)
legend([b1,b2],{'w restricted to ? 0','w allowed to be < 0'});legend boxoff
% set(gcf,'PaperPosition',[1/4,1/4,24,9]);print(gcf,'-dtiff','-r300','ReplyToReply_EyeTrack_Figure_DevianceImprovement')

save('ws_EyeTrack_ReplyToReply')

end