function DivNorm_Analysis_DirRepl
% Divisive Normalization Project
% Experiment: Direct Replication
% 
% This script runs the following analyses:
% - frequentist statistics (t-tests, regression analyses)
% - Small Telescopes analysis (based on Simonsohn, 2015, Psychol Sci)
% - Hierarchical Bayesian analyses (with JAGS)
% - Model comparison between probit and divisive normalization
%
% The following additional files are required to run this script:
% - datasets of the direct replication experiment (in a folder called
% "ReplicationData")
% - get_CI_for_ES.m (for the Small Telescopes approach)
% - LogReg_V3_JAGS.txt (for the hierarchical Bayesian analyses)
% - gs_probit_fast.m (for the model comparison)
% - fs_probit_fast.m (for the model comparison)
% - gs_norm_fast.m (for the model comparison)
% - fs_norm_fast.m (for the model comparison)
% - fminsearchcon.m (for the model comparison)
%
% The following tools/libraries are required to run this script
% - matjags (for the hierarchical Bayesian analyses)
% - parallel computing (for the hierarchical Bayesian analyses - but could
%   also be switched off)
%
% Note that some of these analyses are very time consuming (esp. the
% Bayesian analysis and the model comparison). I recommend going through it
% step by step.
%
% sebastian.gluth@unibas.ch


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Load and prepare data %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datadir = 'ReplicationData/'; %folder in which data is saved
nsubj = 103; %number of subjects
ntrials = 250; %number of trials

%load in data
data = struct;
for s = 1:nsubj
    data.persubj{s} = load([datadir,'CDdata_',int2str(s)]);
end
disp('loading of data completed...')

fxdata.V3 = []; %will be dataset for fixed effects analysis
fxdata.normV3 = []; %will be dataset for fixed effects analysis
normV3cat = 0:.2:1; %distractor categories (as in Figure 5A of Louie et al.)

%within-subject analyses
for s = 1:nsubj
    
    %preparations
    cdata = data.persubj{s};
    BDMdata = cdata.ContextDepData.BDM.data; %data from BDM auction
    AFCdata = cdata.ContextDepData.ChoiceBehav.data; %data from 3-alternative forced choice task
    
    V3 = BDMdata(AFCdata(:,1),6); %value of worst option (distractor)
    V2 = BDMdata(AFCdata(:,2),6); %value of 2nd best option (target 2)
    V1 = BDMdata(AFCdata(:,3),6); %value of best option (target 1)
    normV3 = 2*V3./(V1+V2); %value of distractor "normalized" to value of targets
    
    %descriptive statistics
    descriptives.best_chosen(s) = mean(AFCdata(:,4)==3);
    descriptives.second_best_chosen(s) = mean(AFCdata(:,4)==2);
    descriptives.distractor_chosen(s) = mean(AFCdata(:,4)==1);
    descriptives.RT(s) = mean(mean(AFCdata(:,5)));    
    
    %clean up data: exclude any trials in which targets were not chosen
    V1c = V1((AFCdata(:,4)==2)|(AFCdata(:,4)==3));
    V2c = V2((AFCdata(:,4)==2)|(AFCdata(:,4)==3));
    V3c = V3((AFCdata(:,4)==2)|(AFCdata(:,4)==3));
    normV3c = normV3((AFCdata(:,4)==2)|(AFCdata(:,4)==3));
    Choice = AFCdata((AFCdata(:,4)==2)|(AFCdata(:,4)==3),4)==3; %1 = best chosen, 0 = 2nd best chosen
    RT = AFCdata((AFCdata(:,4)==2)|(AFCdata(:,4)==3),5); %Response Times
    
    %data quality checks
    quality_check.BDMcorrelation(s) = corr(BDMdata(:,4),BDMdata(:,5)); %correlation of 1st and 2nd bid (should be at least .5)
    quality_check.BDMzeros(s) = sum(BDMdata(:,6)<0.05); %number of snacks with average bid = 0 (should not be > 10)
    quality_check.choice_accuracy.absolute(s) = mean(AFCdata(:,4)==3); %absolute p of target 1 choices (should be >33%)
    quality_check.choice_accuracy.relative(s) = mean(Choice); %p of target 1 choices relative to target 2 choices (should be >50%)
    
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
    analysis.RTa.betas(s,:) = glmfit(Xz,RT); %analysis of RT
    %analysis 2b: multiple regression with normV3
    X = [V1c,V2c,normV3c]; %independent variables
    Xz = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1); %standardize independent variables
    analysis.b2.betas(s,:) = glmfit(Xz,Choice,'binomial');
    fxdata.normV3 = [fxdata.normV3;[X,Choice,zeros(length(X),1)+s]]; %prepare analysis 3b (fixed effects multiple regression)
    analysis.RTb.betas(s,:) = glmfit(Xz,RT); %analysis of RT
    
    %for visualization: reproduce Figure 5A from Louie et al., 2013
    for c = 1:length(normV3cat)-1
        ct = (normV3c>=normV3cat(c))&(normV3c<normV3cat(c+1)); %trials belonging to current category
        analysis.Figure5A(s,c) = mean(Choice(ct));
        analysis.Figure5A_ValueDifferenceTargets(s,c) = mean(V1c(ct)-V2c(ct));
    end
    %do the same for V3 instead of normV3
    V3cat = sortrows([sortrows([(1:length(V3))',V3],2),sort(repmat((1:(length(normV3cat)-1))',length(V3)/(length(normV3cat)-1),1))],1)*[0;0;1];
    V3ccat = V3cat((AFCdata(:,4)==2)|(AFCdata(:,4)==3));
    for c = 1:length(normV3cat)-1
        ct = V3ccat==c; %trials belonging to current category
        analysis.Figure5A_V3(s,c) = mean(Choice(ct));
        analysis.Figure5A_ValueDifferenceTargets_V3(s,c) = mean(V1c(ct)-V2c(ct));
    end
        
end
disp('within-subject analyses completed (warning messages come from subject #3)...')

%exclude subject #3, who has only one distractor with bid > 0
inclsubj = [1,2,4:nsubj]; %for random effects analyses
incltrials = fxdata.V3(:,5)~=3; %for fixed effects analyses


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Frequentist statistics %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%analysis 1a and 1b on the group level
%(choice efficiency as a function of high vs. low V3/normV3)
[~,p,~,t] = ttest(analysis.a1(inclsubj,1),analysis.a1(inclsubj,2),'tail','left');
[~,~,ci,~] = ttest(analysis.a1(inclsubj,:));
d = mean(analysis.a1(inclsubj,1)-analysis.a1(inclsubj,2))./std(analysis.a1(inclsubj,1)-analysis.a1(inclsubj,2));
disp(['t-test of high vs. low V3: t(',int2str(t.df),') = ',num2str(t.tstat,3),'; p = ',num2str(p,3),'; d = ',num2str(d,3)])

figure;hold on;title('Direct replication');ylabel('Relative choice accuracy');
xlim([0.5,2.5]);ylim([0.25,0.95])
plot([1,1],ci(:,1),'k-','LineWidth',10);plot([2,2],ci(:,2),'k-','LineWidth',10)
plot(ones(length(inclsubj),1),analysis.a1(inclsubj,1),'.','Color',[.5,.5,.5]);
plot(ones(length(inclsubj),1)+1,analysis.a1(inclsubj,2),'.','Color',[.5,.5,.5])
plot([0.8,1.2],zeros(1,2)+mean(analysis.a1(inclsubj,1)),'-','LineWidth',2,'Color',[.25,1,.25])
plot([1.8,2.2],zeros(1,2)+mean(analysis.a1(inclsubj,2)),'-','LineWidth',2,'Color',[0,.5,0])
set(gca,'XTick',1:2,'XTickLabel',{'high V3','low V3'},'YTick',.3:.2:.9,'FontSize',12)
% set(gcf,'PaperPosition',[1/4,1/4,8,8]);print(gcf,'-dtiff','-r300','Figures/R1/ReplStudy_LowHigh_V3_R1')

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

figure;hold on;title('Direct replication');ylabel('Effect on relative choice accuracy')
xlim([0.5,3.5]);ylim([-1.5,1.5])
plot([0.5,3.5],[0,0],'k-')
plot([1,1],ci(:,2),'k-','LineWidth',10);plot([2,2],ci(:,3),'k-','LineWidth',10);plot([3,3],ci(:,4),'k-','LineWidth',10);
% plot(zeros(length(inclsubj),1)+1,analysis.a2.betas(inclsubj,2),'.','Color',[.5,.5,.5]);
% plot(zeros(length(inclsubj),1)+2,analysis.a2.betas(inclsubj,3),'.','Color',[.5,.5,.5]);
plot(zeros(length(inclsubj),1)+3,analysis.a2.betas(inclsubj,4),'.','Color',[.5,.5,.5]);
plot([0.8,1.2],zeros(1,2)+mean(analysis.a2.betas(inclsubj,2)),'b-','LineWidth',2)
plot([1.8,2.2],zeros(1,2)+mean(analysis.a2.betas(inclsubj,3)),'r-','LineWidth',2)
plot([2.8,3.2],zeros(1,2)+mean(analysis.a2.betas(inclsubj,4)),'g-','LineWidth',2)
set(gca,'XTick',1:3,'XTickLabel',{'V1','V2','V3'},'FontSize',12)
% set(gcf,'PaperPosition',[1/4,1/4,8,8]);print(gcf,'-dtiff','-r300','Figures/R1/ReplStudy_LogReg_V3_R1')

[~,p(2)] = ttest(analysis.b2.betas(inclsubj,2),zeros(size(analysis.b2.betas(inclsubj,2))),'tail','right');
[~,p(3:4)] = ttest(analysis.b2.betas(inclsubj,3:4),zeros(size(analysis.b2.betas(inclsubj,3:4))),'tail','left');
% [~,~,ci,t] = ttest(analysis.b2.betas(inclsubj,:));
d = mean(analysis.b2.betas(inclsubj,:))./std(analysis.b2.betas(inclsubj,:));
disp(['t-test of beta(normV3): t(',int2str(t.df(4)),') = ',num2str(t.tstat(4),3),'; p = ',num2str(p(4),3),'; d = ',num2str(d(4),3)])

%analysis 3a and 3b
%(multiple logistic fixed-effects regression with V3/normV3)
[analysis.a3.B,analysis.a3.DEV,analysis.a3.STATS] = glmfit(fxdata.V3(incltrials,1:3),fxdata.V3(incltrials,4),'binomial');
[analysis.b3.B,analysis.b3.DEV,analysis.b3.STATS] = glmfit(fxdata.normV3(incltrials,1:3),fxdata.normV3(incltrials,4),'binomial');

% reproduce Figure 5A
[~,~,ci,~] = ttest(analysis.Figure5A(inclsubj,:));
figure;
subplot(2,2,1);hold on;xlim([0.5,5.5]);ylim([0.5,0.8])
plot(nanmean(analysis.Figure5A(inclsubj,:)),'k.','MarkerSize',30)
for x = 1:length(normV3cat)-1
    plot([x,x],ci(:,x),'k-','LineWidth',2);
%     plot(zeros(length(inclsubj),1)+x,analysis.Figure5A(inclsubj&(isnan(analysis.Figure5A(inclsubj,c))==0),c),'b.')
end
set(gca,'XTick',0.5:1:5.5,'XTickLabel',{'0','0.2','0.4','0.6','0.8','1'},'YTick',0:.1:1)
xlabel('normV3');ylabel('Relative choice accuracy')
%show value difference of targets for levels of Figure 5A
[~,~,ci,~] = ttest(analysis.Figure5A_ValueDifferenceTargets(inclsubj,:));
subplot(2,2,2);hold on;xlim([0.5,5.5]);ylim([0.25,0.7])
plot(nanmean(analysis.Figure5A_ValueDifferenceTargets(inclsubj,:)),'k.','MarkerSize',30)
for x = 1:length(normV3cat)-1
    plot([x,x],ci(:,x),'k-','LineWidth',2);
%     plot(zeros(length(inclsubj),1)+x,analysis.Figure5A_ValueDifferenceTargets(inclsubj&(isnan(analysis.Figure5A_ValueDifferenceTargets(inclsubj,c))==0),c),'b.')
end
set(gca,'XTick',0.5:1:5.5,'XTickLabel',{'0','0.2','0.4','0.6','0.8','1'},'YTick',0:.1:1)
xlabel('normV3');ylabel('Value difference of targets (V1 - V2)')
%reproduce Figure 5A but with V3 instead of normV3
[~,~,ci,~] = ttest(analysis.Figure5A_V3(inclsubj,:));
% figure;
subplot(2,2,3);hold on;xlim([0.5,5.5]);ylim([0.5,0.8])
plot(nanmean(analysis.Figure5A_V3(inclsubj,:)),'k.','MarkerSize',30)
for x = 1:length(normV3cat)-1
    plot([x,x],ci(:,x),'k-','LineWidth',2);
%     plot(zeros(length(inclsubj),1)+x,analysis.Figure5A_V3(inclsubj&(isnan(analysis.Figure5A_V3(inclsubj,c))==0),c),'b.')
end
set(gca,'XTick',[1,5],'XTickLabel',{'lowest','highest'},'YTick',0:.1:1)
xlabel('V3');ylabel('Relative choice accuracy')
%show value difference of targets for levels of Figure 5A (with V3)
[~,~,ci,~] = ttest(analysis.Figure5A_ValueDifferenceTargets_V3(inclsubj,:));
subplot(2,2,4);hold on;xlim([0.5,5.5]);ylim([0.25,0.7])
plot(nanmean(analysis.Figure5A_ValueDifferenceTargets_V3(inclsubj,:)),'k.','MarkerSize',30)
for x = 1:length(normV3cat)-1
    plot([x,x],ci(:,x),'k-','LineWidth',2);
%     plot(zeros(length(inclsubj),1)+x,analysis.Figure5A_ValueDifferenceTargets_V3(inclsubj&(isnan(analysis.Figure5A_ValueDifferenceTargets_V3(inclsubj,c))==0),c),'b.')
end
set(gca,'XTick',[1,5],'XTickLabel',{'lowest','highest'},'YTick',0:.1:1)
xlabel('V3');ylabel('Value difference of targets (V1 - V2)')
% set(gcf,'PaperPosition',[1/4,1/4,20,20]);print(gcf,'-dtiff','-r300','Figures/ReplStudy_normV3vsV3b')

%%%%%%%%%%%%%%%%%%%
%%% RT analyses %%%
%%%%%%%%%%%%%%%%%%%

%pre-analysis for response times: identify cut-off for RT (> mean + 4*SD)
allRT = zeros(length(inclsubj)*ntrials,1);
for s = 1:length(inclsubj)
    S = inclsubj(s);
    cdata = data.persubj{S};
    AFCdata = cdata.ContextDepData.ChoiceBehav.data; %data from 3-alternative forced choice task
    allRT((s-1)*ntrials+1:s*ntrials) = AFCdata(:,5);
end
RTcut = mean(allRT)+4*std(allRT); %note: min RT is 349 ms, so that's fine
%perform further behavioral analyses (currently: analysis of RT)
for s = 1:nsubj %note: the excluded subject is still analyzed, so make sure to exclude her when running the statistics
    
    %preparations
    cdata = data.persubj{s};
    BDMdata = cdata.ContextDepData.BDM.data; %data from BDM auction
    AFCdata = cdata.ContextDepData.ChoiceBehav.data; %data from 3-alternative forced choice task
    
    V3 = BDMdata(AFCdata(:,1),6); %value of worst option (distractor)
    V2 = BDMdata(AFCdata(:,2),6); %value of 2nd best option (target 2)
    V1 = BDMdata(AFCdata(:,3),6); %value of best option (target 1)
    normV3 = 2*V3./(V1+V2); %value of distractor "normalized" to value of targets
    
    %clean up data: exclude trials with distractor chosen or too slow RT
    include_trials = ((AFCdata(:,4)==2)|(AFCdata(:,4)==3))&(AFCdata(:,5)<=RTcut);
    V1c = V1(include_trials);
    V2c = V2(include_trials);
    V3c = V3(include_trials);
    normV3c = normV3(include_trials);
    RT = AFCdata(include_trials,5)*1000; %Scale up to ms
    
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
figure('OuterPosition',[440 378 600 800]);hold on;title('Direct replication');ylabel('Effect on response time')
xlim([0.5,3.5]);%ylim([0.5,0.8])
plot([0.5,3.5],[0,0],'k-')
for x = 1:size(ci,2)
    plot([x,x],ci(:,x),'k-','LineWidth',10);
    plot(zeros(length(inclsubj),1)+x,analysis.RTa.betas(inclsubj,x+1),'.','Color',[.5,.5,.5])
end
plot([0.8,1.2],zeros(1,2)+nanmean(analysis.RTa.betas(inclsubj,2)),'b-','LineWidth',2)
plot([1.8,2.2],zeros(1,2)+nanmean(analysis.RTa.betas(inclsubj,3)),'r-','LineWidth',2)
plot([2.8,3.2],zeros(1,2)+nanmean(analysis.RTa.betas(inclsubj,4)),'g-','LineWidth',2)
set(gca,'XTick',1:3,'XTickLabel',{'V1 - V2','V1 + V2','V3'},'FontSize',16)
% set(gcf,'PaperPosition',[1/4,1/4,12,12]);print(gcf,'-dtiff','-r300','Figures/R1/ReplStudy_LinReg_RT_R1')

%plot RT as a function of V3
[~,~,ci,~] = ttest(analysis.RTb.visualization_V3(inclsubj,:));
figure;hold on;xlim([0.5,5.5]);ylim([1550,2300])
for x = 1:ncatRT
    plot([x,x],ci(:,x),'k-','LineWidth',6);
    plot([0.8,1.2]+(x-1),zeros(1,2)+nanmean(analysis.RTb.visualization_V3(inclsubj,x)),'g-','LineWidth',3)
end
plot(nanmean(analysis.RTb.visualization_V3(inclsubj,:)),'g-')
set(gca,'XTick',[1,5],'XTickLabel',{'lowest','highest'},'YTick',1600:200:2200,'FontSize',16)
xlabel('V3');ylabel('RT in ms')
% set(gcf,'PaperPosition',[1/4,1/4,12,9]);print(gcf,'-dtiff','-r300','Figures/R1/ReplStudy_RTbyV3')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Small Telescopes analysis %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d33 = -0.2462868; % effect size that would give Louie2013 33% power; done in R: pwr.t.test(n=40,d=NULL,sig.level=.05,power=.33,type="one.sample",alternative="two.sided")
% dsig = -tinv(.95,length(inclsubj)-1)/sqrt(length(inclsubj)); %effect size that would be required to find a siginficant effect

%median-split of low and high V3
es1 = mean(analysis.a1(inclsubj,1)-analysis.a1(inclsubj,2))./std(analysis.a1(inclsubj,1)-analysis.a1(inclsubj,2));
[~,~,~,stats] = ttest(analysis.a1(inclsubj,1),analysis.a1(inclsubj,2));
t = stats.tstat;
df = stats.df;
EffectSize_CIs_LowHigh = get_CI_for_ES(es1,t,df);

%logistic regression of V3
es2 = mean(analysis.a2.betas(inclsubj,4))./std(analysis.a2.betas(inclsubj,4));
[~,~,~,stats] = ttest(analysis.a2.betas(inclsubj,4));
t = stats.tstat;
df = stats.df;
EffectSize_CIs_LogReg = get_CI_for_ES(es2,t,df);

figure;hold on;ylabel('Effect size {\it d}');xlim([0.5,2.5]);ylim([-.3,.3]);xlabel('Statistical test')
plot([0.5,2.5],[0,0],'k-');
p1 = plot([0.75,2.25],[d33,d33],'k--','LineWidth',2);
p2 = plot([1,1],EffectSize_CIs_LowHigh(1,1:2),'k-','LineWidth',4);
p3 = plot([1,1],EffectSize_CIs_LowHigh(1,3:4),'-','LineWidth',4,'Color',[.7,.7,.7]);
plot([2,2],EffectSize_CIs_LogReg(1,1:2),'k-','LineWidth',4);
plot([2,2],EffectSize_CIs_LogReg(1,3:4),'-','LineWidth',4,'Color',[.7,.7,.7]);
plot([0.9,1.1],[es1,es1],'g-','LineWidth',2);plot([1.9,2.1],[es2,es2],'g-','LineWidth',2);
legend([p1,p3,p2],{'{\it d}_3_3_%','90% CI','95% CI'},'Location','Northeast');legend boxoff
set(gca,'XTick',1:2,'XTickLabel',{'High vs. low V3','Logistic regression'},'YTick',-1:.1:1,'FontSize',12);
% set(gcf,'PaperPosition',[1/4,1/4,16,8]);print(gcf,'-dtiff','-r300','Figures/R1/ReplStudy_SmallTelescopes_R1')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Hierarchical Bayesian analyses %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inclsubj = 1:nsubj; %optional: you may include all subjects, because the hierarchical Bayesian analysis should have less issues with the outlier (ID3)
close all
addpath /Applications/MATLAB_ADD/matjags

%specify data inputs for JAGS
jagsdata.V1 = [];
jagsdata.V2 = [];
jagsdata.V3 = [];
jagsdata.normV3 = [];
jagsdata.C = [];
jagsdata.S = [];
for s = 1:length(inclsubj)
    S = inclsubj(s);
    
    %preparations
    cdata = data.persubj{S};
    BDMdata = cdata.ContextDepData.BDM.data; %data from BDM auction
    AFCdata = cdata.ContextDepData.ChoiceBehav.data; %data from 3-alternative forced choice task
    
    V3 = BDMdata(AFCdata(:,1),6); %value of worst option (distractor)
    V2 = BDMdata(AFCdata(:,2),6); %value of 2nd best option (target 2)
    V1 = BDMdata(AFCdata(:,3),6); %value of best option (target 1)
    normV3 = 2*V3./(V1+V2); %value of distractor "normalized" to value of targets
    
    %clean up data: exclude any trials in which targets were not chosen
    jagsdata.V1 = [jagsdata.V1;V1((AFCdata(:,4)==2)|(AFCdata(:,4)==3))];
    jagsdata.V2 = [jagsdata.V2;V2((AFCdata(:,4)==2)|(AFCdata(:,4)==3))];
    jagsdata.V3 = [jagsdata.V3;V3((AFCdata(:,4)==2)|(AFCdata(:,4)==3))];
    jagsdata.normV3 = [jagsdata.normV3;normV3((AFCdata(:,4)==2)|(AFCdata(:,4)==3))];
    jagsdata.C = [jagsdata.C;AFCdata((AFCdata(:,4)==2)|(AFCdata(:,4)==3),4)==3]; %1 = best chosen, 0 = 2nd best chosen
    jagsdata.S = [jagsdata.S;zeros(sum((AFCdata(:,4)==2)|(AFCdata(:,4)==3)),1)+s];
end

% Specify a reasonable prior SD of group-level coefficients
priorSD = 5;

% Assign Matlab Variables to the Observed Nodes
N = size(jagsdata.C,1);
% subj = inclsubj;
subj = 1:length(inclsubj);
jags_data = struct('priorSD',priorSD,'N',N,'subj',subj,'nsubj',length(subj),...
                   'S',jagsdata.S,'C',jagsdata.C,'V1',jagsdata.V1,'V2',jagsdata.V2,'V3',jagsdata.V3);
               
% JAGS settings
doparallel = 1;
nchains = 4;
nburnin = 2000;
nsamples = 10000;
thin = 10;

% Initial Values
clear init0
for iv = 1:nchains
    I.muintercept = (rand-0.5)*priorSD;
    I.muwV1 = (rand-0.5)*priorSD;
    I.muwV2 = (rand-0.5)*priorSD;
    I.muwV3 = (rand-0.5)*priorSD;
    I.sigmaintercept = (rand+0.2)*priorSD;
    I.sigmawV1 = (rand+0.2)*priorSD;
    I.sigmawV2 = (rand+0.2)*priorSD;
    I.sigmawV3 = (rand+0.2)*priorSD;
    init0(iv) = I;
end

%start parallel pool
parpool;

% Use JAGS to Sample
tic
fprintf( 'Running JAGS ...\n' );
[samples_with_V3, stats_with_V3] =  matjags(jags_data,fullfile(pwd,'LogReg_V3_JAGS.txt'),init0, ...
	'doparallel' , doparallel,'nchains', nchains,'nburnin', nburnin,'nsamples', nsamples,'thin',thin,...
	'monitorparams',{'muintercept','muwV1','muwV2','muwV3',...
                     'sigmaintercept','sigmawV1','sigmawV2','sigmawV3',...
                     'intercept','wV1','wV2','wV3'},...
	'savejagsoutput',1,'verbosity',1,'cleanup',0,'workingdir','tmpjags');
toc

%check that all R-hats are below 1.01
allRhats = [stats_with_V3.Rhat.muintercept;stats_with_V3.Rhat.muwV1;stats_with_V3.Rhat.muwV2;stats_with_V3.Rhat.muwV3;...
            stats_with_V3.Rhat.sigmaintercept;stats_with_V3.Rhat.sigmawV1;stats_with_V3.Rhat.sigmawV2;stats_with_V3.Rhat.sigmawV3;...
            stats_with_V3.Rhat.intercept;stats_with_V3.Rhat.wV1;stats_with_V3.Rhat.wV2;stats_with_V3.Rhat.wV3;stats_with_V3.Rhat.deviance];
if max(allRhats) < 1.01
    disp('all R-hats are below 1.01')
else
    disp('WARNING: not all R-hats are below 1.01!')
end

%get Bayes Factors (Savage-Dickey method)
prior = normpdf(0,0,priorSD);
posterior_V1 = normpdf(0,mean(samples_with_V3.muwV1(1:nsamples*nchains)),std(samples_with_V3.muwV1(1:nsamples*nchains)));
BF10.V1 = prior./posterior_V1;
posterior_V2 = normpdf(0,mean(samples_with_V3.muwV2(1:nsamples*nchains)),std(samples_with_V3.muwV2(1:nsamples*nchains)));
BF10.V2 = prior./posterior_V2;
posterior_V3 = normpdf(0,mean(samples_with_V3.muwV3(1:nsamples*nchains)),std(samples_with_V3.muwV3(1:nsamples*nchains)));
BF10.V3 = prior./posterior_V3;

figure;hold on;xlabel('Regression coefficient of V3');ylabel('Frequency');xlim([-0.1,0.1])
histogram(samples_with_V3.muwV3(1:nsamples*nchains),x,'FaceColor','g','EdgeColor','g');
p3 = plot(x,y*nsamples*nchains*normpdf(x,mean(samples_with_V3.muwV3(1:nsamples*nchains)),std(samples_with_V3.muwV3(1:nsamples*nchains))),'LineWidth',3,'Color','g');
histogram(randn(nsamples*nchains,1)*priorSD,x,'FaceColor','k','EdgeColor','k');
p0 = plot(x,y*nsamples*nchains*normpdf(x,0,priorSD),'LineWidth',3,'Color','k');
legend([p0,p3],{'Prior','Posterior'});legend boxoff
set(gca,'XTick',-.1:.05:.1,'YTick',0:1000:3000,'FontSize',12)
% set(gcf,'PaperPosition',[1/4,1/4,16,8]);print(gcf,'-dtiff','-r300','Figures/R1/ReplStudy_BayesianRegression_R1')

% THE SAME BUT WITH normV3 INSTEAD OF V3
jags_data = struct('priorSD',priorSD,'N',N,'subj',subj,'nsubj',length(subj),...
                   'S',jagsdata.S,'C',jagsdata.C,'V1',jagsdata.V1,'V2',jagsdata.V2,'V3',jagsdata.normV3);
               
% Initial Values
clear init0
for iv = 1:nchains
    I.muintercept = (rand-0.5)*priorSD;
    I.muwV1 = (rand-0.5)*priorSD;
    I.muwV2 = (rand-0.5)*priorSD;
    I.muwV3 = (rand-0.5)*priorSD;
    I.sigmaintercept = (rand+0.2)*priorSD;
    I.sigmawV1 = (rand+0.2)*priorSD;
    I.sigmawV2 = (rand+0.2)*priorSD;
    I.sigmawV3 = (rand+0.2)*priorSD;
    init0(iv) = I;
end

% Use JAGS to Sample
tic
fprintf( 'Running JAGS ...\n' );
[samples_with_normV3, stats_with_normV3] =  matjags(jags_data,fullfile(pwd,'LogReg_V3_JAGS.txt'),init0, ...
	'doparallel' , doparallel,'nchains', nchains,'nburnin', nburnin,'nsamples', nsamples,'thin',thin,...
	'monitorparams',{'muintercept','muwV1','muwV2','muwV3',...
                     'sigmaintercept','sigmawV1','sigmawV2','sigmawV3',...
                     'intercept','wV1','wV2','wV3'},...
	'savejagsoutput',1,'verbosity',1,'cleanup',0,'workingdir','tmpjags');
toc

%check that all R-hats are below 1.01
allRhats = [stats_with_normV3.Rhat.muintercept;stats_with_normV3.Rhat.muwV1;stats_with_normV3.Rhat.muwV2;stats_with_normV3.Rhat.muwV3;...
            stats_with_normV3.Rhat.sigmaintercept;stats_with_normV3.Rhat.sigmawV1;stats_with_normV3.Rhat.sigmawV2;stats_with_normV3.Rhat.sigmawV3;...
            stats_with_normV3.Rhat.intercept;stats_with_normV3.Rhat.wV1;stats_with_normV3.Rhat.wV2;stats_with_normV3.Rhat.wV3;stats_with_normV3.Rhat.deviance];
if max(allRhats) < 1.01
    disp('all R-hats are below 1.01')
else
    disp('WARNING: not all R-hats are below 1.01!')
end

%get Bayes Factors (Savage-Dickey method)
prior = normpdf(0,0,priorSD);
posterior_V1 = normpdf(0,mean(samples_with_normV3.muwV1(1:nsamples*nchains)),std(samples_with_normV3.muwV1(1:nsamples*nchains)));
BF10.V1_with_normV3 = prior./posterior_V1;
posterior_V2 = normpdf(0,mean(samples_with_normV3.muwV2(1:nsamples*nchains)),std(samples_with_normV3.muwV2(1:nsamples*nchains)));
BF10.V2_with_normV3 = prior./posterior_V2;
posterior_V3 = normpdf(0,mean(samples_with_normV3.muwV3(1:nsamples*nchains)),std(samples_with_normV3.muwV3(1:nsamples*nchains)));
BF10.normV3 = prior./posterior_V3;


%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%% Model comparison %%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

disp('done with behavioral analyses, starting model fitting...') %this message is provided because from here on, things take VERY long...
close all

%general settings
fei = [2000,1000]; %max number of function evaluations and iterations
fs = [1,3,5,1]; %number of fminsearches with best parameter estimates from grid search; random pick of best 30%/1%; random pick; nested (or random again)

%probit model settings
probit.gs = 10; %number of gridsearches per parameter for probit model
probit.params_minmax = [eps;30]; %the free parameter of this model is the standard deviation of the normally distributed "choice error"
probit.params_range = linspace(0.1,18.1,probit.gs); %grid search parameter range for probit model

%normalization model settings
norm.gs = probit.gs+zeros(1,5); %number of gridsearches per parameter for normalization model
norm.params_minmax = [0,eps,0,eps,0;200,200,10,30,10];%gain, semisaturation, weight term, fixed noise, scaled noise
norm.prepr = [1,1,0,0.1,0;181,181,1.8,18.1,1.8]; %preparation of grid search parameter range for normalization model
for x = 1:length(norm.prepr)
    norm.params_range(x,:) = linspace(norm.prepr(1,x),norm.prepr(2,x),norm.gs(x));
end

for s = 1:nsubj
%     tic
    %preparations
    cdata = data.persubj{s};
    BDMdata = cdata.ContextDepData.BDM.data; %data from BDM auction
    AFCdata = cdata.ContextDepData.ChoiceBehav.data; %data from 3-alternative forced choice task
    
    V3 = BDMdata(AFCdata(:,1),6); %value of worst option (distractor)
    V2 = BDMdata(AFCdata(:,2),6); %value of 2nd best option (target 2)
    V1 = BDMdata(AFCdata(:,3),6); %value of best option (target 1)
    C = AFCdata(:,4); %3 = best chosen, 2 = 2nd best chosen, 1 = distractor chosen
    Vcu = [V3.*(C==1)+V2.*(C==2)+V1.*(C==3),V1.*(C==1)+V1.*(C==2)+V2.*(C==3),V2.*(C==1)+V3.*(C==2)+V3.*(C==3)]; %value of chosen, better unchosen and worse unchosen option
    ntrials = length(C);
    
    %probit model, grid search
    deviance = gs_probit_fast(Vcu,ntrials,probit.params_range,probit.gs);
    probit.dev_gs(s,:) = deviance;
    %fminsearch
    preparams = zeros(sum(fs),1);
	predev = zeros(sum(fs),1);
	best_gs = sortrows([deviance,(1:probit.gs)'])*[0;1];
    for x = 1:sum(fs)
        if x <= fs(1) %start with best
            params_init = probit.params_range(best_gs(1));
        elseif (x > fs(1)) && (x <= sum(fs(1:2))) %start with good
            params_init = probit.params_range(best_gs(randi(round(probit.gs/3))+1));
        else %start with random
            params_init = probit.params_range(randi(probit.gs));
        end
        [preparams(x),predev(x)] = fs_probit_fast(Vcu,params_init,probit.params_minmax,fei);
    end
    probit.params(s) = preparams(find(predev==min(predev),1));
    probit.dev_fs(s) = predev(find(predev==min(predev),1));
    
	%normalization model, grid search
    deviance = gs_norm_fast(Vcu,ntrials,norm.params_range,norm.gs(1));
    norm.dev_gs{s} = deviance;
	%fminsearch
    preparams = zeros(sum(fs),length(norm.params_minmax));
	predev = zeros(sum(fs),1);
	preexitflag = zeros(sum(fs),1);
    thresh_deviance = max(((1:prod(norm.gs))==round(prod(norm.gs)*.01)).*sort(reshape(deviance,prod(norm.gs),1))');
	[D1,D2,D3,D4,D5] = ind2sub(size(deviance),find(deviance<=thresh_deviance)); %best 1%
	[D1b,D2b,D3b,D4b,D5b] = ind2sub(size(deviance),find(deviance==min(min(min(min(min(deviance))))))); %best
    for x = 1:sum(fs)
        if x <= fs(1) %start with best
            r = randi(length(D1b)); %in case of multiple best values
            params_init = [norm.params_range(1,D1b(r)),norm.params_range(2,D2b(r)),norm.params_range(3,D3b(r)),norm.params_range(4,D4b(r)),norm.params_range(5,D5b(r))];
        elseif (x > fs(1)) && (x <= sum(fs(1:2))) %start with good
            r = randi(length(D1));
            params_init = [norm.params_range(1,D1(r)),norm.params_range(2,D2(r)),norm.params_range(3,D3(r)),norm.params_range(4,D4(r)),norm.params_range(5,D5(r))];
        elseif (x > sum(fs(1:2))) && (x <= sum(fs(1:3))) %start with random
            params_init = [norm.params_range(1,randi(norm.gs(1))),norm.params_range(2,randi(norm.gs(2))),norm.params_range(3,randi(norm.gs(3))),norm.params_range(4,randi(norm.gs(4))),norm.params_range(5,randi(norm.gs(5)))];
        else %start with best of probit
            params_init = [100,50,0,probit.params(s),0];
        end
        [preparams(x,:),predev(x),preexitflag(x)] = fs_norm_fast(Vcu,params_init,norm.params_minmax,fei);
    end
	norm.params(s,:) = preparams(find(predev==min(predev),1),:);
    norm.dev_fs(s) = predev(find(predev==min(predev),1));
	norm.exitflag(s) = preexitflag(find(predev==min(predev),1));
	norm.allexitflags(s,:) = preexitflag;
    
    save(['ws_DivNorm_Analsis_DirRepl_with_Modeling_gs',int2str(probit.gs)]) %commented here, but it makes sense to save after each subject in case something happens after a long time
    disp(['done model fitting of subject #',int2str(s),' ...'])
%     toc
end

%figure showing the fit improvement from probit to norm in each subject
figure('OuterPosition',[440 378 1920/2 1080/2]);hold on;
bar(sort(probit.dev_fs(inclsubj)-norm.dev_fs(inclsubj),'descend'),'FaceColor','k');
plot([0,length(inclsubj)+1],zeros(1,2)+chi2inv(.95,4),'k--')
text(length(inclsubj)*.9,chi2inv(.95,4)*.9,'{\it p} < .05')
xlabel('Participants');ylabel('Improvement of deviance','FontSize',14)
% set(gcf,'PaperPosition',[1/4,1/4,32,8]);print(gcf,'-dtiff','-r300','Figures/R1/ReplStudy_ProbitNormalization_R1')

end