# Divisive Normalization Project
# Experiment: Eye Tracking
# 
# This script runs the eye-tracking analyses including:
# - preprocessing (reading in data from BeGaze and EventDetection and processing this data for analyses)
# - bottom-up effects (of value on first fixations)
# - top-down effects (i.e., development of fixations within trial as a function of value)
# - path analysis (i.e., testing whether the distractor effect is mediated by value-based attention)
# - the "Krajbich effect" (i.e., an influence of gaze on choice)
# - preparation of datasets for Matlab (to generate figures in there)
#
# The following additional files are required to run this script:
# - behavioral datasets of the eye-tracking experiment (in a folder called "Behavior")
# - eye-tracking datasets (in a folder called "Eye")
#
# The following libraries are required to run this script
# - readtext (for reading in the data from EventDetection files)
# - car (for ANOVAs)
# - heplots (for eta-squared of ANOVAs)
# - sjstats (for eta-squared of an alternative way of performing ANOVAs based on "aov")
# - lavaan (for the path analysis)
#
# Note that reading and preprocessing of the eye-tracking data takes a couple of minutes
#
# sebastian.gluth@unibas.ch


### ----------- ###
### 1. Settings ###
### ----------- ###

# clear workspace and figures
rm(list=ls())
graphics.off()

# load required libraries 
library(readtext)
library(car)
library(heplots)
library(sjstats)
library(lavaan)

#set working directory and specify subjects
base_directory <- "/Volumes/sgluth/Desktop/normalization/Masterproject/Analysis/EyeTracking/Main/"
behavdata_directory <- "Behavior/"
eyedata_directory <- "Eye/"
IDs <- c(1:11,13:19,21,23:37,39:45)
n <- length(IDs)
setwd(base_directory)


### -------------------------------------------- ###
### 2. Get, Check and Preprocess Behavioral Data ###
### -------------------------------------------- ###

#load behavioral data, screen behavioral data for quality
bdata <- list()
bdmdata <- list()
excl_crit <- matrix(ncol = 8,nrow = n)
ntrials <- vector(length = n)
for(s in 1:n){
  S <- IDs[s]
  bdata[[S]] <- read.table(paste0(base_directory,behavdata_directory,"ID",S,"/choice_output"))
  colnames(bdata[[S]]) <- c("item1","item2","item3","distractor","difficulty","rating1","rating2","rating3","position1","position2","position3","choice","RT")
  #IMPORTANT: generate new variables rating1c and rating2c because it's not always that rating1 > rating2!
  bdata[[S]]$rating1c <- bdata[[S]]$rating1*(bdata[[S]]$rating1>=bdata[[S]]$rating2)+bdata[[S]]$rating2*(bdata[[S]]$rating1<bdata[[S]]$rating2)
  bdata[[S]]$rating2c <- bdata[[S]]$rating2*(bdata[[S]]$rating1>=bdata[[S]]$rating2)+bdata[[S]]$rating1*(bdata[[S]]$rating1<bdata[[S]]$rating2)
  bdata[[S]]$rating3c <- bdata[[S]]$rating3 #for the sake of completeness
  #IMPORTANT: do the same with positions
  bdata[[S]]$position1c <- bdata[[S]]$position1*(bdata[[S]]$rating1>=bdata[[S]]$rating2)+bdata[[S]]$position2*(bdata[[S]]$rating1<bdata[[S]]$rating2)
  bdata[[S]]$position2c <- bdata[[S]]$position2*(bdata[[S]]$rating1>=bdata[[S]]$rating2)+bdata[[S]]$position1*(bdata[[S]]$rating1<bdata[[S]]$rating2)
  bdata[[S]]$position3c <- bdata[[S]]$position3 #for the sake of completeness
  
  #check for exclusion criteria
  #1. The correlation of subjective ratings across the first and second rating trials is ≥ .50
  bdmdata[[S]] <- read.table(paste0(base_directory,behavdata_directory,"ID",IDs[s],"/bdm_run"))
  colnames(bdmdata[[S]]) <- c("run","trial","snackID","rating","rt")
  ordered_bdmmat <- bdmdata[[S]][order(bdmdata[[S]]$snackID),]
  excl_crit[s,1] <- cor(ordered_bdmmat$rating[seq(1,nrow(ordered_bdmmat),2)],ordered_bdmmat$rating[seq(2,nrow(ordered_bdmmat),2)])
  excl_crit[s,5] <- excl_crit[s,1]>=.5
  #2. The absolute choice accuracy (i.e., the probability of choosing the best option) is ≥ 1/3
  #create new variable ("choice123") which indicates whether best (1), 2nd best (2), or worst (3) option has been chosen
  bdata[[S]]$choice123 <- (bdata[[S]]$choice==bdata[[S]]$position1c)+
                          (bdata[[S]]$choice==bdata[[S]]$position2c)*2+
                          (bdata[[S]]$choice==bdata[[S]]$position3c)*3
  excl_crit[s,2] <- mean(bdata[[S]]$choice123==1)
  excl_crit[s,6] <- excl_crit[s,2]>=(1/3)
  #3. Choice efficiency (i.e., the probability choosing the better out of the two target options) is ≥ 1/2
  excl_crit[s,3] <- mean(bdata[[S]]$choice123[bdata[[S]]$choice123<=2]==1)
  excl_crit[s,7] <- excl_crit[s,3]>=(1/2)
  #4. More than 40% of snacks were rated with 0
  excl_crit[s,4] <- mean(((ordered_bdmmat$rating[seq(1,nrow(ordered_bdmmat),2)]+ordered_bdmmat$rating[seq(2,nrow(ordered_bdmmat),2)])/2)<.01)
  excl_crit[s,8] <- excl_crit[s,4]<=.4
}
#update IDs and number of subjects according to exclusion criteria
realIDs <- IDs[rowSums(excl_crit[,5:8])==4]
n <- length(realIDs)
#for the moment, exclude ID 11 because BeGaze did not provide full dataset
#realIDs <- realIDs[realIDs!=11]
#n <- length(realIDs)
ntrials <- rep(nrow(bdata[[S]]),n)


### --------------------------------------- ###
### 3. Get and Preprocess Eye-Tracking Data ###
### --------------------------------------- ###

###
#load eye-tracking data via BeGaze file
eyefile <- paste(base_directory,eyedata_directory,"Event_Statistics_Single_NormalizationStudy_noheader.txt",sep = "")
eyedata <- read.delim(file = eyefile,sep = ",")

#split data by subjects
eyeID <- levels(eyedata$Participant)
split_eyeID <- strsplit(eyeID,"_")
edata <- list()
for (s in 1:n){
  S <- realIDs[s]
  #find matching ID name
  cmatch <- 0
  for (t in 1:length(eyeID)){
    cmatch <- cmatch+(split_eyeID[[t]][length(split_eyeID[[t]])]==paste0("ID",S))*t #note: only works if filenames end with "_IDxx"
  }
  edata[[S]] <- eyedata[eyedata$Participant==eyeID[cmatch],] 
}

#reduce data so that only fixations within AOIs from choice phase remain
fixdata <- list()
for (s in 1:n){
  S <- realIDs[s]
  fixdata[[S]] <- edata[[S]]
  #look up every event whether it happend during choice phase; add trial numbers
  fixdata[[S]][,(ncol(fixdata[[S]])+1):(ncol(fixdata[[S]])+3)] <- NA
  trialonset <- 0
  trialnr <- 1
  inphase <- 0
  for (t in 1:nrow(fixdata[[S]])){
    fixdata[[S]][t,ncol(fixdata[[S]])-2] <- fixdata[[S]]$Event.Start.Trial.Time..ms.[t]-trialonset
    fixdata[[S]][t,ncol(fixdata[[S]])-1] <- trialnr
    fixdata[[S]][t,ncol(fixdata[[S]])] <- inphase
    if (fixdata[[S]]$Content[t]=="ChoicePhase"){
      inphase <- 1
      trialonset <- fixdata[[S]]$Event.Start.Trial.Time..ms.[t]
    }
    if ((fixdata[[S]]$Content[t]=="ChoiceLogged_1")||(fixdata[[S]]$Content[t]=="ChoiceLogged_2")||(fixdata[[S]]$Content[t]=="ChoiceLogged_3")){
      fixdata[[S]][t,ncol(fixdata[[S]])] <- 0
      inphase <- 0
      trialnr <- trialnr + 1
    }
  }
  #delete events that happend outside choice phase
  fixdata[[S]] <- fixdata[[S]][fixdata[[S]][,ncol(fixdata[[S]])]==1,]
  #delete all events that are not fixations within AOIs; also delete last column
  fixdata[[S]] <- fixdata[[S]][(fixdata[[S]]$AOI.Name!="-")&(fixdata[[S]]$AOI.Name!="White Space")&
                               (fixdata[[S]]$Category=="Fixation"),1:(ncol(fixdata[[S]])-1)]
  #rename last two columns
  names(fixdata[[S]])[names(fixdata[[S]])==paste("V",ncol(fixdata[[S]])-1,sep="")] <- "intrialonset"
  names(fixdata[[S]])[names(fixdata[[S]])==paste("V",ncol(fixdata[[S]]),sep="")] <- "trialnr"
}

###
#load eye-tracking data via event file (only required for subject #11, but set up for all subjects to allow comparison)

#required: define Areas of Interest (order: L, U, R)
AOInames <- c("Option L","Option U","Option R")
AOI <- matrix(data = c(463,569.5,863,969.5,760,110.5,1160,510.5,1057,569.5,1457,969.5),nrow = 3,ncol = 4,byrow = T)
rownames(AOI) <- AOInames
colnames(AOI) <- c("x1","y1","x2","y2")

#loop over subjects
replace_eye <- 11 #specify the subject(s) for which BeGaze data should be replaced by event file data (set to "realIDs" to replace all)
for (s in 1:n){
  S = realIDs[s]
  #read in event file using the readtext package
  #eventfile_name <- paste("EventDetection/EyeTrackingData_ID",S," Events.txt",sep="")
  eventfile_name <- paste("EventDetectionOnlyRightEye/EyeTrackingData_ID",S," Events.txt",sep="")
  eventfile <- paste(base_directory,eyedata_directory,eventfile_name,sep = "")
  ef_raw <- readtext(eventfile)
  #pre-process the file to get isolate the following events: UserEvent, Fixation 
  ef_txt <- ef_raw$text
  ef_split_n <-strsplit(ef_txt,"\n") #separate lines
  nentries <- length(ef_split_n[[1]]) #number of lines in the event files
  #look at User Events to get start and end of choice phase
  choice_se <- matrix(nrow = ntrials,ncol = 2) #will be start and and of choice phase
  ctrials <- 1 #trial counter
  x <- 1
  while (ctrials <= ntrials[s]){
    c_entry <- ef_split_n[[1]][x]
    match1 <- grepl("UserEvent",c_entry)
    match1b <- grepl("ChoicePhase",c_entry)
    match1c <- any(c(grepl("ChoiceLogged_1",c_entry),grepl("ChoiceLogged_2",c_entry),grepl("ChoiceLogged_3",c_entry)))
    if (match1b == T){
      c_entry_split_t <- strsplit(c_entry,"\t") #separate tab-stopped entries within line
      choice_se[ctrials,1] <- char2num(c_entry_split_t[[1]][4])
    }
    if (match1c == T){
      c_entry_split_t <- strsplit(c_entry,"\t") #separate tab-stopped entries within line
      choice_se[ctrials,2] <- char2num(c_entry_split_t[[1]][4])
      ctrials = ctrials + 1
    }
    x = x + 1
  }
  #look at the remaining events to get fixations
  fixation_info <- matrix(NA,nrow = nentries-x, ncol = 8)
  cfixations <- 1
  for (y in x:nentries){
    c_entry <- ef_split_n[[1]][y]
    match2 <- grepl("Fixation R",c_entry) #only "Right" fixations, because data is saved twice (once for "Left", once for "Right")
    if (match2 == T){
      c_entry_split_t <- strsplit(c_entry,"\t") #separate tab-stopped entries within line
      fix_start <- char2num(c_entry_split_t[[1]][4]) #start of fixation
      inphase <- (fix_start>=choice_se[,1])&(fix_start<choice_se[,2]) #check whether fixation happened within a choice phase
      fix_XY <- c(char2num(c_entry_split_t[[1]][7]),char2num(c_entry_split_t[[1]][8]))
      inAOI <- (fix_XY[1]>AOI[,1])&(fix_XY[1]<AOI[,3])&(fix_XY[2]>AOI[,2])&(fix_XY[2]<AOI[,4])#check whether fixation happened within an AOI
      #save fixation information of fixation happened during choice and within an AOI
      if (any(inphase) && any(inAOI)){
        fix_end <- char2num(c_entry_split_t[[1]][5]) #end of fixation
        fix_dur <- char2num(c_entry_split_t[[1]][6]) #duration of fixation
        fix_AOI <- AOInames[which(inAOI)]
        fix_trial <- which(inphase)
        fixation_info[cfixations,] <- c(fix_start/1000,fix_end/1000,fix_dur/1000,fix_XY,fix_AOI,(fix_start-choice_se[fix_trial,1])/1000,fix_trial)
        cfixations <- cfixations + 1
      }
    }
  }
  fixation_info <- fixation_info[1:(cfixations-1),] #reduce the matrix
  #replace current subjects "fixdata" by fixation_info
  if (any(S==replace_eye)){
    fixdata[[S]] <- data.frame(fixation_info)
    colnames(fixdata[[S]]) <- c("Event.Start.Raw.Time..ms.","Event.End.Raw.Time..ms.","Event.Duration.Trial.Time..ms.",
                                "Fixation.Position.X..px.","Fixation.Position.Y..px.","AOI.Name","intrialonset","trialnr")
  }
}

#create a new data set in which the last fixation of each trial is deleted
fixdata_nlf <- list()#nlf is for "no last fixation"
for (s in 1:n){
  S = realIDs[s]
  exclude_fix <-c(fixdata[[S]]$trialnr[2:length(fixdata[[S]]$trialnr)]!=fixdata[[S]]$trialnr[1:(length(fixdata[[S]]$trialnr)-1)],TRUE)
  fixdata_nlf[[S]] <- fixdata[[S]][exclude_fix==FALSE,]
}

#create another dataset with trial-wise information
trialdata <- list()
trialdata_nlf <- list()
for (s in 1:n){
  S = realIDs[s]
  f <- rep(NA,ntrials[s])
  trialdata[[S]] <- data.frame(nfixL = f, nfixU = f, nfixR = f, dgazeL = f, dgazeU = f, dgazeR = f, firstfix = f, row.names = 1:ntrials[s])
  for (t in 1:ntrials[s]){
    stdata <- fixdata[[S]][fixdata[[S]]$trialnr==t,] #single-trial data
    trialdata[[S]]$nfixL[t] <- sum(stdata$AOI.Name=="Option L")
    trialdata[[S]]$nfixU[t] <- sum(stdata$AOI.Name=="Option U")
    trialdata[[S]]$nfixR[t] <- sum(stdata$AOI.Name=="Option R")
    stduration <- as.numeric(paste(stdata$Event.Duration.Trial.Time..ms.)) #necessary because data is saved as factors
    trialdata[[S]]$dgazeL[t] <- sum((stdata$AOI.Name=="Option L")*stduration)
    trialdata[[S]]$dgazeU[t] <- sum((stdata$AOI.Name=="Option U")*stduration)
    trialdata[[S]]$dgazeR[t] <- sum((stdata$AOI.Name=="Option R")*stduration)
    trialdata[[S]]$firstfix[t] <- (stdata$AOI.Name[1]=="Option L")+2*(stdata$AOI.Name[1]=="Option U")+3*(stdata$AOI.Name[1]=="Option R")
  }
  #add total and relative fixation durations
  trialdata[[S]]$totaldur <- trialdata[[S]]$dgazeL+trialdata[[S]]$dgazeU+trialdata[[S]]$dgazeR
  trialdata[[S]]$reldurL <- trialdata[[S]]$dgazeL/trialdata[[S]]$totaldur
  trialdata[[S]]$reldurU <- trialdata[[S]]$dgazeU/trialdata[[S]]$totaldur
  trialdata[[S]]$reldurR <- trialdata[[S]]$dgazeR/trialdata[[S]]$totaldur
  
  #do the same but without the last fixation in each trial
  trialdata_nlf[[S]] <- data.frame(nfixL = f, nfixU = f, nfixR = f, dgazeL = f, dgazeU = f, dgazeR = f, firstfix = f, row.names = 1:ntrials[s])
  for (t in 1:ntrials[s]){
    stdata <- fixdata_nlf[[S]][fixdata_nlf[[S]]$trialnr==t,] #single-trial data
    trialdata_nlf[[S]]$nfixL[t] <- sum(stdata$AOI.Name=="Option L")
    trialdata_nlf[[S]]$nfixU[t] <- sum(stdata$AOI.Name=="Option U")
    trialdata_nlf[[S]]$nfixR[t] <- sum(stdata$AOI.Name=="Option R")
    stduration <- as.numeric(paste(stdata$Event.Duration.Trial.Time..ms.)) #necessary because data is saved as factors
    trialdata_nlf[[S]]$dgazeL[t] <- sum((stdata$AOI.Name=="Option L")*stduration)
    trialdata_nlf[[S]]$dgazeU[t] <- sum((stdata$AOI.Name=="Option U")*stduration)
    trialdata_nlf[[S]]$dgazeR[t] <- sum((stdata$AOI.Name=="Option R")*stduration)
    trialdata_nlf[[S]]$firstfix[t] <- (stdata$AOI.Name[1]=="Option L")+2*(stdata$AOI.Name[1]=="Option U")+3*(stdata$AOI.Name[1]=="Option R")
  }
  #add total and relative fixation durations
  trialdata_nlf[[S]]$totaldur <- trialdata_nlf[[S]]$dgazeL+trialdata_nlf[[S]]$dgazeU+trialdata_nlf[[S]]$dgazeR
  trialdata_nlf[[S]]$reldurL <- trialdata_nlf[[S]]$dgazeL/trialdata_nlf[[S]]$totaldur
  trialdata_nlf[[S]]$reldurU <- trialdata_nlf[[S]]$dgazeU/trialdata_nlf[[S]]$totaldur
  trialdata_nlf[[S]]$reldurR <- trialdata_nlf[[S]]$dgazeR/trialdata_nlf[[S]]$totaldur
}


### ------------------------------------------- ###
### 4. Eye-tracking analysis 1: first fixations ###
### ------------------------------------------- ###

###does the value of the distractor influences the likelihood that it is fixated first in a trial?
###also test the overall likelhood to fixate best, second-best and worst first in a trial
#NOTE: FOR THIS ANALYSIS, THE LAST FIXATION IN A TRIAL SHOULD BE EXCLUDED
betas_firstfix <- matrix(ncol = 2,nrow = n)
freq_first_123 <- matrix(ncol = 3,nrow = n)
for (s in 1:n){
  S <- realIDs[s]
  #recode first fixation to whether first fixation was made on distractor or not
  firstfixation_on_3 <- (trialdata_nlf[[S]]$firstfix==bdata[[S]]$position3c)
  value_of_3 <- bdata[[S]]$rating3c
  glm_fix <- glm(firstfixation_on_3 ~ value_of_3, family = binomial)
  betas_firstfix[s,] <- glm_fix$coefficients
  #best vs. second-best vs. worst
  firstfixation_on_1 <- (trialdata_nlf[[S]]$firstfix==bdata[[S]]$position1c)
  firstfixation_on_2 <- (trialdata_nlf[[S]]$firstfix==bdata[[S]]$position2c)
  freq_first_123[s,] <- c(mean(firstfixation_on_1,na.rm = T),mean(firstfixation_on_2,na.rm = T),mean(firstfixation_on_3,na.rm = T))
}
#### .... #### .... #### .... #### .... STATISTICAL ANALYSIS #### .... #### .... #### .... #### .... #### ....

#Effect of distractor value on probability of fixating the distractor
stats_V3_on_First <- t.test(betas_firstfix[,2],alternative = "greater")
d_interact_V3_on_First <- mean(betas_firstfix[,2])/sd(betas_firstfix[,2])

#Anova for all participants
anmat <- freq_first_123
wf <- expand.grid(factor(1:3))
names(wf) <- c("Option")
anova_b.lm <- lm(anmat~1)
anova_freq_first <- Anova(anova_b.lm,idesign=~Option,idata=wf,type="III",multivariate=FALSE)
partial_eta_sq_anova_freq_first <- etasq(anova_freq_first)

#alternative ANOVA analysis (with "aov") for analysis without outlier (because the car-package has a weird problem here(???))
Subjects <- as.factor(rep(1:36,3))
Options <- as.factor(c(rep(1,36),rep(2,36),rep(3,36)))
AV <- c(freq_first_123[c(1:9,11:37),1],freq_first_123[c(1:9,11:37),2],freq_first_123[c(1:9,11:37),3])
aovDataset <- data.frame(Subjects,Options,AV)
ANOVA <- aov(AV ~ Options+Error(Subjects),aovDataset)
partial_eta_sq_ANOVA <- eta_sq(ANOVA)
summary(ANOVA)


### ---------------------------------------------------- ###
### 5. Eye-tracking analysis 2: development of fixations ###
### ---------------------------------------------------- ###

###change in frequency of looking at best, 2nd-best, worst changes within each trial
#NOTE: FOR THIS ANALYSIS, THE LAST FIXATION IN A TRIAL SHOULD BE EXCLUDED
nfixtime1 <- list()
nfixtime2 <- list()
nfixtime3 <- list()
nfixt1 <- matrix(ncol = 100,nrow = n) #this will be a matrix of consequtive fixations per trial (ncol: choose a high number of possible within-trial fixations)
nfixt2 <- matrix(ncol = 100,nrow = n) #this will be a matrix of consequtive fixations per trial (ncol: choose a high number of possible within-trial fixations)
nfixt3 <- matrix(ncol = 100,nrow = n) #this will be a matrix of consequtive fixations per trial (ncol: choose a high number of possible within-trial fixations)
nfixt3_l <- matrix(ncol = 100,nrow = n) #this will be a matrix of consequtive fixations per trial (ncol: choose a high number of possible within-trial fixations)
nfixt3_h <- matrix(ncol = 100,nrow = n) #this will be a matrix of consequtive fixations per trial (ncol: choose a high number of possible within-trial fixations)
betas_nfix <- matrix(ncol = 4,nrow = n) #betas of logistic regression testing whether fixations on D decrease and even more so for low-value D
inclFixN <- c() #required for the logistic regression
colnames(betas_nfix) <- c("intercept","V3","#fixation","interaction")
for (s in 1:n){
  S <- realIDs[s]
  nfixtime1[[S]] <- matrix(ncol = 100,nrow = ntrials) #this will be a matrix of consequtive fixations per trial (ncol: choose a high number of possible within-trial fixations)
  nfixtime2[[S]] <- matrix(ncol = 100,nrow = ntrials) #this will be a matrix of consequtive fixations per trial (ncol: choose a high number of possible within-trial fixations)
  nfixtime3[[S]] <- matrix(ncol = 100,nrow = ntrials) #this will be a matrix of consequtive fixations per trial (ncol: choose a high number of possible within-trial fixations)
  for (t in 1:ntrials[s]){
    cdata <- fixdata_nlf[[S]][fixdata_nlf[[S]]$trialnr==t,]
    if (nrow(cdata)>1){
      nfixtime1[[S]][t,1:nrow(cdata)] <- (bdata[[S]]$position1c[t]==1)&(cdata$AOI.Name=="Option L")|
                                         (bdata[[S]]$position1c[t]==2)&(cdata$AOI.Name=="Option U")|
                                         (bdata[[S]]$position1c[t]==3)&(cdata$AOI.Name=="Option R")
      nfixtime2[[S]][t,1:nrow(cdata)] <- (bdata[[S]]$position2c[t]==1)&(cdata$AOI.Name=="Option L")|
                                         (bdata[[S]]$position2c[t]==2)&(cdata$AOI.Name=="Option U")|
                                         (bdata[[S]]$position2c[t]==3)&(cdata$AOI.Name=="Option R")
      nfixtime3[[S]][t,1:nrow(cdata)] <- (bdata[[S]]$position3c[t]==1)&(cdata$AOI.Name=="Option L")|
                                         (bdata[[S]]$position3c[t]==2)&(cdata$AOI.Name=="Option U")|
                                         (bdata[[S]]$position3c[t]==3)&(cdata$AOI.Name=="Option R")
    }
  }
  nfixt1[s,] <- colMeans(nfixtime1[[S]],na.rm = T)
  nfixt2[s,] <- colMeans(nfixtime2[[S]],na.rm = T)
  nfixt3[s,] <- colMeans(nfixtime3[[S]],na.rm = T)
  #split trials into low- vs. high-value distractors
  high_d <- bdata[[S]]$rating3c>median(bdata[[S]]$rating3c)
  nfixt3_l[s,] <- colMeans(nfixtime3[[S]][high_d==0,],na.rm = T)
  nfixt3_h[s,] <- colMeans(nfixtime3[[S]][high_d==1,],na.rm = T)
  #multiple logistic regression: does distractor fixation decrease with time and is this modulated by value?
  tV3 <- c() #will be value of distractor (predictor)
  tFixN <- c() #will be fixation number (predictor)
  tFix3 <- c() #will be whether distractor was fixated (dependent variable)
  #for the logistic regression, include only those consecutive within-trial fixation numbers for which at least 30 fixations have been recorded
  inclFixN[s] <- sum(colSums(is.na(nfixtime1[[S]])==0)>=30)
  for (t in 1:ntrials[s]){
    nfix <- min(sum(is.na(nfixtime1[[S]][t,])==0),inclFixN[s]) #number of fixations in current trial
    if ((nfix > 0) && (bdata[[S]]$choice123[t]!=3)){ #only update if the current trial had at least one fixation and option 3 was not chosen
      tV3 <- c(tV3,rep(bdata[[S]]$rating3c[t],nfix))
      tFixN <- c(tFixN,1:nfix)
      tFix3 <- c(tFix3,nfixtime3[[S]][t,1:nfix])
    }
  }
  tV3 <- (tV3-mean(tV3))/sd(tV3) #standardize before generating the interaction term
  tFixN <- (tFixN-mean(tFixN))/sd(tFixN) #standardize before generating the interaction term
  tV3_x_tFixN <- tV3*tFixN
  glm_fix <- glm(tFix3 ~ tV3 + tFixN + tV3_x_tFixN, family = binomial)
  betas_nfix[s,] <- glm_fix$coefficients
}
#### .... #### .... #### .... #### .... STATISTICAL ANALYSIS #### .... #### .... #### .... #### .... #### ....

#ANOVA with factors Options and Fixation Number: the critical result is the interaction
max_nfix <- min(sum(colSums(is.na(nfixt1))==0),sum(colSums(is.na(nfixt2))==0),sum(colSums(is.na(nfixt3))==0))
anmat <- cbind(nfixt1[,1:max_nfix],nfixt2[,1:max_nfix],nfixt3[,1:max_nfix])
wf <- expand.grid(factor(1:7),factor(1:3))
names(wf) <- c("FixationNumber","Option")
anova_b.lm <- lm(anmat~1)
anova_fix_development <- Anova(anova_b.lm,idesign=~FixationNumber*Option,idata=wf,type="III",multivariate=FALSE)
partial_eta_sq_anova_fix_development <- etasq(anova_fix_development)

#Interaction effect of distractor value and fixation number
stats_interact_V3_FixNumb <- t.test(betas_nfix[,4],alternative = "greater")
d_interact_V3_FixNumb <- mean(betas_nfix[,4])/sd(betas_nfix[,4])


### ----------------------------------------- ###
### 6. Eye-tracking analysis 3: path analyses ###
### ----------------------------------------- ###
###path analysis 1: higher value of D leads to higher relative fixation duration on D, which leads to lower relative choice efficiency
###path analysis 2: higher value of D leads to higher relative fixation duration on D, which leads to lower absolute choice efficiency (like in Gluth et al., 2018, eLife)
coef_path1_D_Deye <- c()
coef_path1_D_RCA <- c()
coef_path1_Deye_RCA <- c()
coef_path2_D_Deye <- c()
coef_path2_D_ACA <- c()
coef_path2_Deye_ACA <- c()
for (s in 1:n){
  S <- realIDs[s]
  
  #specify relative fixation duration of distractor
  trialdata[[S]]$reldur3 <- trialdata[[S]]$reldurL*(bdata[[S]]$position3c==1)+trialdata[[S]]$reldurU*(bdata[[S]]$position3c==2)+trialdata[[S]]$reldurR*(bdata[[S]]$position3c==3)

  #specify all required variables
  D1 <- bdata[[S]]$rating3c[bdata[[S]]$choice123<3]
  Deye1 <- trialdata[[S]]$reldur3[bdata[[S]]$choice123<3]
  RCA <- bdata[[S]]$choice123[bdata[[S]]$choice123<3]==1
  D2 <- bdata[[S]]$rating3c
  Deye2 <- trialdata[[S]]$reldur3
  ACA <- bdata[[S]]$choice123==1
  
  #clean up (take out trials without gaze info)
  D1 <- D1[is.na(Deye1)==F]
  RCA <- RCA[is.na(Deye1)==F]
  Deye1 <- Deye1[is.na(Deye1)==F]
  D2 <- D2[is.na(Deye2)==F]
  ACA <- ACA[is.na(Deye2)==F]
  Deye2 <- Deye2[is.na(Deye2)==F]
  
  #standardize independent variables
  D1 <- (D1-mean(D1))/sd(D1)
  Deye1 <- (Deye1-mean(Deye1))/sd(Deye1)
  PAmat1 <- as.data.frame(matrix(c(D1,Deye1,RCA),nrow = length(D1),ncol = 3))
  names(PAmat1) <- c("D","Deye","RCA")
  D2 <- (D2-mean(D2))/sd(D2)
  Deye2 <- (Deye2-mean(Deye2))/sd(Deye2)
  PAmat2 <- as.data.frame(matrix(c(D2,Deye2,ACA),nrow = length(D2),ncol = 3))
  names(PAmat2) <- c("D","Deye","ACA")  
  #specify and run the path analysis
  model.one <-  'Deye ~ D
                 RCA ~ D + Deye'
  fit1 <- sem(model.one,data = PAmat1)
  cfit1 <- coef(fit1)
  coef_path1_D_Deye[s] <- cfit1[1]
  coef_path1_D_RCA[s] <- cfit1[2]
  coef_path1_Deye_RCA[s] <- cfit1[3]
  model.two <-  'Deye ~ D
                 ACA ~ D + Deye'
  fit2 <- sem(model.two,data = PAmat2)
  cfit2 <- coef(fit2)
  coef_path2_D_Deye[s] <- cfit2[1]
  coef_path2_D_ACA[s] <- cfit2[2]
  coef_path2_Deye_ACA[s] <- cfit2[3]
}
#Stats for path coefficients
stats_path1_D_Deye <- t.test(coef_path1_D_Deye,alternative = "greater")
d_path1_D_Deye <- mean(coef_path1_D_Deye)/sd(coef_path1_D_Deye)
stats_path1_Deye_RCA <- t.test(coef_path1_Deye_RCA,alternative = "less")
d_path1_Deye_RCA <- mean(coef_path1_Deye_RCA)/sd(coef_path1_Deye_RCA)
stats_path1_D_RCA <- t.test(coef_path1_D_RCA,alternative = "less")
d_path1_D_RCA <- mean(coef_path1_D_RCA)/sd(coef_path1_D_RCA)


### ------------------------------------------------------------------------------------ ###
### 7. Eye-tracking analysis 4: influence of attention on choice (the "Krajbich effect") ###
### ------------------------------------------------------------------------------------ ###

#final time advantage and choice probability (excluding the last fixation)
betas_Krajbich <- matrix(ncol = 3,nrow = n)
colnames(betas_Krajbich) <- c("intercept","value","gaze")
reldurcat <- (0:5)/5
Pleft_FTA <- matrix(ncol = (length(reldurcat)-1), nrow = n)
for (s in 1:n){
  S <- realIDs[s]
  
  #specify relative fixation duration per option 
  trialdata_nlf[[S]]$reldur1 <- trialdata_nlf[[S]]$reldurL*(bdata[[S]]$position1c==1)+trialdata_nlf[[S]]$reldurU*(bdata[[S]]$position1c==2)+trialdata_nlf[[S]]$reldurR*(bdata[[S]]$position1c==3)
  trialdata_nlf[[S]]$reldur2 <- trialdata_nlf[[S]]$reldurL*(bdata[[S]]$position2c==1)+trialdata_nlf[[S]]$reldurU*(bdata[[S]]$position2c==2)+trialdata_nlf[[S]]$reldurR*(bdata[[S]]$position2c==3)
  trialdata_nlf[[S]]$reldur3 <- trialdata_nlf[[S]]$reldurL*(bdata[[S]]$position3c==1)+trialdata_nlf[[S]]$reldurU*(bdata[[S]]$position3c==2)+trialdata_nlf[[S]]$reldurR*(bdata[[S]]$position3c==3)
  
  #logistic regression: whether an item is chosen as a function of its relative value and its relative gaze duration
  relative_value <- c(bdata[[S]]$rating1c-0.5*(bdata[[S]]$rating2c+bdata[[S]]$rating3c),
                      bdata[[S]]$rating2c-0.5*(bdata[[S]]$rating1c+bdata[[S]]$rating3c),
                      bdata[[S]]$rating3c-0.5*(bdata[[S]]$rating1c+bdata[[S]]$rating2c))
  relative_gaze <- c(trialdata_nlf[[S]]$reldur1,trialdata_nlf[[S]]$reldur2,trialdata_nlf[[S]]$reldur3)
  AV <- c(bdata[[S]]$choice123==1,bdata[[S]]$choice123==2,bdata[[S]]$choice123==3)
  glm_Krajbich <- glm(AV ~ relative_value + relative_gaze, family = binomial)
  betas_Krajbich[s,] <- glm_Krajbich$coefficients
  
}
#Stats
stats_gaze_on_choice <- t.test(betas_Krajbich[,3])
d_gaze_on_choice <- mean(betas_Krajbich[,3])/sd(betas_Krajbich[,3])


### --------------------------------- ###
### 8. Create Output Files for Matlab ###
### --------------------------------- ###
fixdata_for_Matlab <- data.frame()
behavdata_for_Matlab <- data.frame()
for (s in 1:n){
  S <- realIDs[s]
  fixdata_for_Matlab <- rbind(fixdata_for_Matlab,data.frame(as.numeric(paste(fixdata[[S]]$Event.Duration.Trial.Time..ms.)),
                                                            as.numeric(paste(fixdata[[S]]$intrialonset)),
                                                            as.numeric(fixdata[[S]]$AOI.Name)+any(S==replace_eye),
                                                            as.numeric(fixdata[[S]]$trialnr),S))
  
  behavdata_for_Matlab <- rbind(behavdata_for_Matlab,data.frame(bdata[[S]],1:ntrials[s],S))
}
write.csv(fixdata_for_Matlab,"fixdata_for_Matlab",row.names = FALSE)
write.csv(behavdata_for_Matlab,"behavdata_for_Matlab",row.names = FALSE)